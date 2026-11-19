#!/usr/bin/env python3
"""박스가 구워진 추출 이미지에서 YOLO txt 라벨을 복원한다.

추출 이미지(`<코드><갯수><위치>_<원본명>.jpg`)에 그려진 사각형 테두리를 색으로
찾아내 원본 이미지 옆에 txt 를 다시 만든다. 엑셀의 결함 갯수와 대조해 검증한다.

  - 테두리 색은 get_color_for_class 고정 팔레트 → 클래스를 특정할 수 있다
  - 박스 내부는 마스크의 '구멍' 이라 라벨 글자 칩(채워진 사각형)과 구분된다
  - 테두리는 선 굵기의 절반만큼 안쪽으로 파고들므로 그만큼 되돌려 보정한다

기본은 미리보기이고 --apply 를 줘야 txt 를 쓴다.
"""
import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPT_DIR))

# get_color_for_class 와 같은 팔레트 (RGB)
PALETTE = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (208, 177, 52), (255, 0, 255),
           (0, 255, 255), (255, 165, 0), (128, 0, 128), (0, 128, 128), (128, 128, 0)]

NAME_RE = re.compile(r"^([a-z]+)(\d{2})([UMB])_(.+)$")


def line_width(w: int, h: int) -> int:
    """추출 당시 쓰인 테두리 굵기 (review_gui 와 동일 식)."""
    return max(2, round(min(w, h) / 500))


def detect_boxes(img, old_cls: int, expect: int, tol: int = 60):
    """엑셀이 알려준 개수(expect)만큼 큰 사각형을 찾는다.

    반환: (박스 목록, 분리비). 분리비 = expect 번째 구멍 면적 / 그 다음 구멍 면적.
    글자 구멍보다 박스 내부가 훨씬 크므로 이 비가 크면 신뢰할 수 있다.
    """
    h, w = img.shape[:2]
    r, g, b = PALETTE[old_cls % len(PALETTE)]
    target = np.array([b, g, r], dtype=np.int16)          # BGR
    mask = (np.abs(img.astype(np.int16) - target).sum(axis=2) < tol).astype(np.uint8)
    if not mask.any():
        return [], 0.0

    # 박스 내부 = 마스크의 구멍. 라벨 글자 구멍도 같이 잡히므로 면적순으로 고른다
    contours, hier = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hier is None:
        return [], 0.0
    holes = []
    for cnt, hr in zip(contours, hier[0]):
        if hr[3] < 0:                                     # 바깥 윤곽은 건너뜀
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        holes.append((bw * bh, x, y, bw, bh))
    holes.sort(reverse=True)
    if len(holes) < expect:
        return [], 0.0

    chosen = holes[:expect]
    nxt = holes[expect][0] if len(holes) > expect else 0
    ratio = (chosen[-1][0] / nxt) if nxt else float("inf")

    half = line_width(w, h) / 2                           # 테두리 굵기 절반 보정
    boxes = []
    for _, x, y, bw, bh in chosen:
        boxes.append((max(0.0, x - half), max(0.0, y - half),
                      min(float(w), x + bw + half), min(float(h), y + bh + half)))
    boxes.sort(key=lambda b: (b[1], b[0]))
    return boxes, ratio


def to_yolo(box, w: int, h: int):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2 / w, (y1 + y2) / 2 / h, (x2 - x1) / w, (y2 - y1) / h)


def load_excel(xlsx: Path):
    """{색인 파일명: (결함 코드, 결함 갯수, Defect Type, 원본 파일명)}"""
    from openpyxl import load_workbook
    ws = load_workbook(xlsx, data_only=True)["결함 기록"]
    out = {}
    for r in range(4, ws.max_row + 1, 3):
        code, en, cnt = (ws.cell(row=r, column=c).value for c in (4, 6, 7))
        for col in (10, 11, 12):
            orig = ws.cell(row=r + 1, column=col).value
            idx  = ws.cell(row=r + 2, column=col).value
            if idx:
                out[str(idx)] = (code, int(cnt or 0), en, str(orig))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("boxed", type=Path, help="박스 구워진 추출 이미지 폴더")
    ap.add_argument("originals", type=Path, help="원본 이미지 폴더 (txt 를 여기에 쓴다)")
    ap.add_argument("--excel", type=Path, required=True, help="대조할 엑셀")
    ap.add_argument("--config", type=Path, default=SCRIPT_DIR / "config_defect_canada")
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    y = yaml.safe_load((args.config / "classes.yaml").read_text(encoding="utf-8"))
    new_id = {y["names"][i]: i for i in range(y["nc"])}
    old_names = [l.strip() for l in
                 (args.config / "obj.names.bak").read_text(encoding="utf-8").splitlines()
                 if l.strip()]
    old_id = {n: i for i, n in enumerate(old_names)}

    excel = load_excel(args.excel)
    boxed = sorted(args.boxed.glob("*.jpg")) + sorted(args.boxed.glob("*.png"))
    print(f"추출 이미지 {len(boxed)}장 / 엑셀 기록 {len(excel)}건\n")

    per_orig = defaultdict(list)     # 원본 stem → [(새 클래스, yolo...)]
    ok = mismatch = missing = 0
    rows = []
    for f in boxed:
        m = NAME_RE.match(f.stem)
        if not m:
            print(f"  ⚠ 이름 형식 밖: {f.name}"); continue
        code, cnt_s, pos, orig_stem = m.groups()
        meta = excel.get(f.name)
        if meta is None:
            print(f"  ⚠ 엑셀에 없음: {f.name}"); missing += 1; continue
        _, exp_cnt, en, orig_name = meta
        if en not in old_id or en not in new_id:
            print(f"  ⚠ 클래스 이름 미상: {en}"); continue

        img = cv2.imread(str(f))
        if img is None:
            print(f"  ⚠ 읽기 실패: {f.name}"); continue
        h, w = img.shape[:2]
        found, ratio = detect_boxes(img, old_id[en], exp_cnt)
        status = "OK" if len(found) == exp_cnt and ratio >= 2 else "확인필요"
        if status == "OK": ok += 1
        else: mismatch += 1
        rows.append((f.name, code, en, exp_cnt, len(found), ratio, status, w, h))

        for bx in found:
            per_orig[orig_name].append((new_id[en],) + to_yolo(bx, w, h))

    print(f"{'색인 파일명':<48}{'코드':<5}{'엑셀':>5}{'검출':>5}{'분리비':>8}  판정")
    print("-" * 84)
    for name, code, en, exp, got, ratio, st, w, h in rows:
        mark = "✅" if st == "OK" else "⚠️"
        rs = "∞" if ratio == float("inf") else f"{ratio:.1f}"
        print(f"{name[:46]:<48}{code:<5}{exp:>5}{got:>5}{rs:>8}  {mark}")
    print("-" * 84)
    print(f"일치 {ok} / 확인필요 {mismatch} / 엑셀 누락 {missing}")

    originals = sorted(p for p in args.originals.iterdir()
                       if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})
    labeled = {o for o in per_orig}
    print(f"\n원본 {len(originals)}장 중 박스 있는 {len(labeled)}장, "
          f"빈 txt {len(originals) - len(labeled)}장")

    if not args.apply:
        print("\n미리보기입니다. 쓰려면 --apply 를 붙이세요.")
        return
    if mismatch:
        sys.exit("\n확인이 필요한 건이 있어 중단합니다.")

    written = 0
    for p in originals:
        lines = [f"{c} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"
                 for c, cx, cy, bw, bh in per_orig.get(p.name, [])]
        p.with_suffix(".txt").write_text(
            ("\n".join(lines) + "\n") if lines else "", encoding="utf-8")
        written += 1
    print(f"\n{written}개 txt 작성 (새 yaml 번호 기준)")


if __name__ == "__main__":
    main()
