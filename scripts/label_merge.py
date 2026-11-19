#!/usr/bin/env python3
"""label_merge.py — YOLO 라벨 폴더 비교 / 병합 (git diff · git merge 방식)

두 라벨 폴더(ours = 기존 버전, theirs = 새 버전)를 이미지 단위로 비교하고,
박스를 IoU로 짝지어 무엇이 추가·삭제·변경됐는지 보여준다.
base(공통 조상)를 주면 git처럼 3-way 병합을 한다.

사용법
------
  # 1) 차이 보기 (git diff --stat 처럼)
  python scripts/label_merge.py diff  OURS THEIRS
  python scripts/label_merge.py diff  OURS THEIRS -v          # 박스 단위 상세
  python scripts/label_merge.py diff  OURS THEIRS --csv diff.csv

  # 2) 병합 (2-way): base 없이 두 버전만 합치기
  python scripts/label_merge.py merge OURS THEIRS --out MERGED --strategy union

  # 3) 병합 (3-way): base 대비 각자 바꾼 것을 합치기 (git merge 와 동일한 개념)
  python scripts/label_merge.py merge OURS THEIRS --base BASE --out MERGED

폴더 구조는 하위 폴더째로 비교한다 (예: 1110subcam/xxx.txt). 이미지 폴더를 넣어도
txt 파일만 읽는다. 입력 폴더는 절대 수정하지 않고, 결과는 --out 에만 쓴다.

diff 표기
---------
  파일:  M 변경  A theirs에만 있음  D ours에만 있음  C 충돌(merge)
  박스:  - ours에만   + theirs에만   ~ 같은 위치인데 클래스가 다름

박스 짝짓기
-----------
  IoU >= --iou(기본 0.5) 인 박스끼리 같은 물체로 본다.
  같은 클래스면 "같음", 다른 클래스면 "클래스 변경(~)".

병합 규칙
---------
  2-way (--base 없음)
    union  : 공통 박스 + ours에만 + theirs에만 모두 유지 (기본값)
    ours   : ours 그대로 (theirs에서는 아무것도 안 가져옴)
    theirs : theirs 그대로
    클래스가 다른 박스(~)는 충돌로 기록하고 --conflict 쪽 클래스를 쓴다.

  3-way (--base 있음)
    - 한쪽만 바꾼 것(추가·삭제·클래스 변경)은 그대로 반영
    - 양쪽이 같은 박스를 서로 다르게 바꾸면 충돌
      (예: ours는 클래스 변경, theirs는 삭제)
    - 충돌은 --conflict(기본 ours) 쪽을 택하고 conflicts.csv 에 남긴다.

결과물 (--out 폴더)
-------------------
  병합된 txt 전부 (하위 폴더 구조 유지)
  _merge_report.csv   파일별 요약
  _conflicts.csv      충돌 목록 (없으면 생성 안 함)
"""

from __future__ import annotations

import argparse
import csv
import signal
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ══════════════════════════════════════════════════════════════════════════════
# 데이터
# ══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class Box:
    cls: int
    cx: float
    cy: float
    w: float
    h: float
    raw: str  # 원본 줄 (출력 시 그대로 사용해 정밀도 보존)

    def iou(self, o: "Box") -> float:
        ix = max(0.0, min(self.cx + self.w / 2, o.cx + o.w / 2) - max(self.cx - self.w / 2, o.cx - o.w / 2))
        iy = max(0.0, min(self.cy + self.h / 2, o.cy + o.h / 2) - max(self.cy - self.h / 2, o.cy - o.h / 2))
        inter = ix * iy
        union = self.w * self.h + o.w * o.h - inter
        return inter / union if union > 0 else 0.0

    def short(self, names: Optional[List[str]] = None) -> str:
        c = names[self.cls] if names and 0 <= self.cls < len(names) else str(self.cls)
        return f"{c:<13} cx={self.cx:.3f} cy={self.cy:.3f} w={self.w:.3f} h={self.h:.3f}"


def read_boxes(p: Optional[Path]) -> List[Box]:
    if p is None or not p.exists():
        return []
    out = []
    for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
        t = line.split()
        if len(t) < 5:
            continue
        try:
            out.append(Box(int(float(t[0])), *map(float, t[1:5]), raw=" ".join(t[:5])))
        except ValueError:
            continue
    return out


def nkey(rel: str) -> str:
    """맥 한글 파일명(NFD/NFC) 차이를 무시하기 위한 키."""
    return unicodedata.normalize("NFC", rel)


def scan(folder: Optional[Path]) -> Dict[str, Path]:
    if folder is None:
        return {}
    res = {}
    for p in folder.rglob("*.txt"):
        rel = p.relative_to(folder).as_posix()
        if any(part.startswith(("_", ".")) for part in Path(rel).parts):
            continue  # _merge_report.csv 같은 산출물 / 숨김 폴더 제외
        if p.name in ("classes.txt", "obj.names"):
            continue
        res[nkey(rel)] = p
    return res


# ══════════════════════════════════════════════════════════════════════════════
# 박스 짝짓기
# ══════════════════════════════════════════════════════════════════════════════


def match(a: List[Box], b: List[Box], thr: float) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """IoU 큰 순서로 그리디 매칭. (짝 목록, a만, b만) 인덱스 반환.
    같은 클래스 쌍을 먼저 잡고, 남은 것 중에서 클래스가 다른 쌍을 잡는다."""
    pairs, used_a, used_b = [], set(), set()
    for same_cls in (True, False):
        cand = []
        for i, x in enumerate(a):
            if i in used_a:
                continue
            for j, y in enumerate(b):
                if j in used_b or (x.cls == y.cls) != same_cls:
                    continue
                v = x.iou(y)
                if v >= thr:
                    cand.append((v, i, j))
        for _, i, j in sorted(cand, reverse=True):
            if i not in used_a and j not in used_b:
                pairs.append((i, j))
                used_a.add(i)
                used_b.add(j)
    return pairs, [i for i in range(len(a)) if i not in used_a], [j for j in range(len(b)) if j not in used_b]


@dataclass
class FileDiff:
    same: List[Tuple[Box, Box]]
    changed: List[Tuple[Box, Box]]  # 위치 같고 클래스 다름
    only_a: List[Box]
    only_b: List[Box]

    @property
    def dirty(self) -> bool:
        return bool(self.changed or self.only_a or self.only_b)


def diff_boxes(a: List[Box], b: List[Box], thr: float) -> FileDiff:
    pairs, oa, ob = match(a, b, thr)
    same = [(a[i], b[j]) for i, j in pairs if a[i].cls == b[j].cls]
    changed = [(a[i], b[j]) for i, j in pairs if a[i].cls != b[j].cls]
    return FileDiff(same, changed, [a[i] for i in oa], [b[j] for j in ob])


# ══════════════════════════════════════════════════════════════════════════════
# diff
# ══════════════════════════════════════════════════════════════════════════════


def cmd_diff(args) -> int:
    ours, theirs = Path(args.ours), Path(args.theirs)
    names = load_names(args.names)
    A, B = scan(ours), scan(theirs)
    keys = sorted(set(A) | set(B))
    rows = []
    tot = dict(M=0, A=0, D=0, same=0, plus=0, minus=0, tilde=0)

    for k in keys:
        a, b = read_boxes(A.get(k)), read_boxes(B.get(k))
        d = diff_boxes(a, b, args.iou)
        if k not in A:
            st = "A"
        elif k not in B:
            st = "D"
        elif d.dirty:
            st = "M"
        else:
            tot["same"] += 1
            continue
        tot[st] += 1
        tot["plus"] += len(d.only_b)
        tot["minus"] += len(d.only_a)
        tot["tilde"] += len(d.changed)
        rows.append((st, k, len(a), len(b), len(d.only_b), len(d.only_a), len(d.changed)))

        print(f"{st} {k:<60} +{len(d.only_b):<3} -{len(d.only_a):<3} ~{len(d.changed):<3} ({len(a)} → {len(b)})")
        if args.verbose:
            for x in d.only_a:
                print(f"    - {x.short(names)}")
            for y in d.only_b:
                print(f"    + {y.short(names)}")
            for x, y in d.changed:
                print(f"    ~ {cname(x.cls, names)} → {cname(y.cls, names)}   iou={x.iou(y):.2f}  {y.short(names)}")

    print("─" * 80)
    print(f"파일 {len(keys)}개: 변경 M {tot['M']} / theirs에만 A {tot['A']} / ours에만 D {tot['D']} / 동일 {tot['same']}")
    print(f"박스: +{tot['plus']} (theirs에만)  -{tot['minus']} (ours에만)  ~{tot['tilde']} (클래스 변경)")

    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["status", "file", "ours_boxes", "theirs_boxes", "plus", "minus", "class_changed"])
            w.writerows(rows)
        print(f"CSV 저장: {args.csv}")
    return 0


# ══════════════════════════════════════════════════════════════════════════════
# merge
# ══════════════════════════════════════════════════════════════════════════════


def merge_2way(a: List[Box], b: List[Box], thr: float, strategy: str, prefer: str):
    d = diff_boxes(a, b, thr)
    conflicts = [("class", x, y) for x, y in d.changed]
    if strategy == "ours":
        return list(a), [], d
    if strategy == "theirs":
        return list(b), [], d
    out = [x for x, _ in d.same]
    out += [(x if prefer == "ours" else y) for x, y in d.changed]
    out += d.only_a + d.only_b
    return out, conflicts, d


def merge_3way(base: List[Box], a: List[Box], b: List[Box], thr: float, prefer: str):
    """git 3-way merge 와 같은 규칙을 박스 단위로 적용."""
    pa, _, new_a = match(base, a, thr)
    pb, _, new_b = match(base, b, thr)
    in_a = {i: a[j] for i, j in pa}  # base 박스 i 가 ours 에서 어떻게 됐는지 (없으면 삭제)
    in_b = {i: b[j] for i, j in pb}
    out, conflicts = [], []

    for i, x in enumerate(base):
        ya, yb = in_a.get(i), in_b.get(i)
        ch_a = ya is None or ya.cls != x.cls  # ours 가 바꿨나
        ch_b = yb is None or yb.cls != x.cls
        if not ch_a and not ch_b:
            out.append(ya)  # 둘 다 그대로 → ours 좌표
        elif ch_a and not ch_b:
            if ya is not None:
                out.append(ya)  # ours 만 클래스 변경
            # ours 만 삭제 → 삭제 반영
        elif ch_b and not ch_a:
            if yb is not None:
                out.append(yb)
        else:  # 양쪽 다 바꿈
            if ya is None and yb is None:
                continue  # 둘 다 삭제 → 삭제
            if ya is not None and yb is not None and ya.cls == yb.cls:
                out.append(ya)  # 같은 클래스로 바꿈 → 충돌 아님
                continue
            pick = ya if prefer == "ours" else yb
            kind = "class" if (ya is not None and yb is not None) else (
                "ours_changed_theirs_deleted" if ya is not None else "ours_deleted_theirs_changed")
            conflicts.append((kind, ya, yb, x))
            if pick is not None:
                out.append(pick)

    # 각자 새로 추가한 박스: 서로 겹치면 하나만
    na, nb = [a[j] for j in new_a], [b[j] for j in new_b]
    d = diff_boxes(na, nb, thr)
    out += [x for x, _ in d.same]
    for x, y in d.changed:
        conflicts.append(("both_added_diff_class", x, y, None))
        out.append(x if prefer == "ours" else y)
    out += d.only_a + d.only_b
    return out, conflicts


def cmd_merge(args) -> int:
    ours, theirs = Path(args.ours).resolve(), Path(args.theirs).resolve()
    base = Path(args.base).resolve() if args.base else None
    out_dir = Path(args.out).resolve()
    for p in (ours, theirs, base):
        if p is not None and (out_dir == p or p in out_dir.parents):
            sys.exit(f"--out 은 입력 폴더({p})와 달라야 합니다.")
    if out_dir.exists() and any(out_dir.iterdir()) and not args.force:
        sys.exit(f"--out 폴더가 비어 있지 않습니다: {out_dir}  (덮어쓰려면 --force)")
    names = load_names(args.names)

    A, B, O = scan(ours), scan(theirs), scan(base)
    keys = sorted(set(A) | set(B))
    report, conflict_rows = [], []
    tot = dict(files=0, written=0, conflict_files=0, conflicts=0, boxes=0)

    for k in keys:
        a, b = read_boxes(A.get(k)), read_boxes(B.get(k))
        if base is not None:
            merged, confs = merge_3way(read_boxes(O.get(k)), a, b, args.iou, args.conflict)
        else:
            merged, confs, _ = merge_2way(a, b, args.iou, args.strategy, args.conflict)

        # 원래 파일명 표기 유지 (ours 우선)
        src = A.get(k) or B.get(k)
        rel = src.relative_to(ours if k in A else theirs)
        dst = out_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text("".join(x.raw + "\n" for x in merged), encoding="utf-8")

        tot["files"] += 1
        tot["written"] += 1
        tot["boxes"] += len(merged)
        status = "C" if confs else ("M" if [x.raw for x in merged] != [x.raw for x in a] else "=")
        if confs:
            tot["conflict_files"] += 1
            tot["conflicts"] += len(confs)
            for c in confs:
                kind, ya, yb = c[0], c[1], c[2]
                conflict_rows.append([
                    k, kind,
                    cname(ya.cls, names) if ya else "(삭제)",
                    cname(yb.cls, names) if yb else "(삭제)",
                    args.conflict,
                    (ya or yb).raw,
                ])
            if args.verbose:
                print(f"C {k}")
                for c in confs:
                    ya, yb = c[1], c[2]
                    print(f"    ! {c[0]:<28} ours={cname(ya.cls, names) if ya else '(삭제)'}  "
                          f"theirs={cname(yb.cls, names) if yb else '(삭제)'}  → {args.conflict}")
        report.append([status, k, len(a), len(b), len(merged), len(confs)])

    with open(out_dir / "_merge_report.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["status", "file", "ours_boxes", "theirs_boxes", "merged_boxes", "conflicts"])
        w.writerows(report)
    if conflict_rows:
        with open(out_dir / "_conflicts.csv", "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["file", "kind", "ours", "theirs", "picked", "box"])
            w.writerows(conflict_rows)

    mode = f"3-way (base={base})" if base else f"2-way (strategy={args.strategy})"
    changed = sum(1 for r in report if r[0] != "=")
    print("─" * 80)
    print(f"병합 방식: {mode}, IoU {args.iou}, 충돌 시 {args.conflict} 우선")
    print(f"파일 {tot['files']}개 저장 → {out_dir}")
    print(f"ours 대비 바뀐 파일 {changed}개, 충돌 파일 {tot['conflict_files']}개 (충돌 {tot['conflicts']}건)")
    print(f"병합 후 박스 {tot['boxes']}개")
    if conflict_rows:
        print(f"충돌 목록: {out_dir / '_conflicts.csv'}")
    return 0


# ══════════════════════════════════════════════════════════════════════════════
# 공통
# ══════════════════════════════════════════════════════════════════════════════


def load_names(p: Optional[str]) -> Optional[List[str]]:
    if not p:
        return None
    return [l.strip() for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]


def cname(c: int, names: Optional[List[str]]) -> str:
    return names[c] if names and 0 <= c < len(names) else str(c)


def main():
    if hasattr(signal, "SIGPIPE"):  # `| head` 로 끊어도 에러 안 나게
        signal.signal(signal.SIGPIPE, signal.SIG_DFL)
    ap = argparse.ArgumentParser(description="YOLO 라벨 폴더 비교/병합 (git diff·merge 방식)",
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("ours", help="기존 버전 라벨 폴더")
        p.add_argument("theirs", help="새 버전 라벨 폴더")
        p.add_argument("--iou", type=float, default=0.5, help="같은 박스로 볼 IoU 기준 (기본 0.5)")
        p.add_argument("--names", help="클래스 이름 파일 (예: config_indonesia/obj.names)")
        p.add_argument("-v", "--verbose", action="store_true", help="박스 단위 상세 출력")

    d = sub.add_parser("diff", help="두 폴더 차이 보기")
    common(d)
    d.add_argument("--csv", help="파일별 차이를 CSV로 저장")

    m = sub.add_parser("merge", help="두 폴더 병합")
    common(m)
    m.add_argument("--out", required=True, help="병합 결과 폴더 (입력 폴더와 달라야 함)")
    m.add_argument("--base", help="공통 조상 폴더 (주면 3-way 병합)")
    m.add_argument("--strategy", choices=["union", "ours", "theirs"], default="union",
                   help="2-way 병합 방식 (기본 union, --base 있으면 무시)")
    m.add_argument("--conflict", choices=["ours", "theirs"], default="ours",
                   help="충돌 시 어느 쪽을 택할지 (기본 ours)")
    m.add_argument("--force", action="store_true", help="--out 폴더가 비어 있지 않아도 진행")

    args = ap.parse_args()
    sys.exit(cmd_diff(args) if args.cmd == "diff" else cmd_merge(args))


if __name__ == "__main__":
    main()
