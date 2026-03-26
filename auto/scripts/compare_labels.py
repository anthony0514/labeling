"""
라벨 비교 스크립트
자동 라벨링 결과(predict_v3/labels)와 원본 라벨(source/안재현_라벨링_500_5)을 비교합니다.
"""
import argparse
from pathlib import Path
from collections import defaultdict, Counter

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def parse_args():
    parser = argparse.ArgumentParser(description="자동/원본 라벨 비교")
    parser.add_argument("--auto", type=str,
                        default=str(PROJECT_ROOT / "runs/detect/predict_v3/labels"),
                        help="자동 라벨 폴더 경로")
    parser.add_argument("--src", type=str,
                        default=str(Path.home() / "labeling/source/안재현_라벨링_500_5"),
                        help="원본 라벨 폴더 경로")
    parser.add_argument("--iou", type=float, default=0.5,
                        help="매칭 IoU 임계값 (기본 0.5)")
    parser.add_argument("--detail", action="store_true",
                        help="파일별 상세 비교 출력")
    return parser.parse_args()


# ── IoU 계산 (YOLO cx cy w h 형식) ──────────────────────────────────────────
def xywh_to_xyxy(cx, cy, w, h):
    return cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2


def iou(b1, b2):
    x1 = max(b1[0], b2[0]); y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2]); y2 = min(b1[3], b2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter == 0:
        return 0.0
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    return inter / (a1 + a2 - inter)


# ── 파일 파싱 ─────────────────────────────────────────────────────────────────
def load_labels(path: Path):
    """[(cls, cx, cy, w, h), ...] 반환. 빈 파일이면 []."""
    boxes = []
    if not path.exists():
        return boxes
    for line in path.read_text().strip().splitlines():
        parts = line.split()
        if len(parts) == 5:
            boxes.append((int(parts[0]), *map(float, parts[1:])))
    return boxes


# ── 파일 단위 매칭 (헝가리안 없이 greedy, 라벨 비교 목적으로 충분) ──────────
def match_boxes(gt_boxes, pred_boxes, iou_thresh):
    matched_gt, matched_pred = set(), set()
    tp = 0
    for gi, gt in enumerate(gt_boxes):
        gt_cls, *gt_xywh = gt
        best_iou, best_pi = 0, -1
        for pi, pred in enumerate(pred_boxes):
            if pi in matched_pred:
                continue
            pred_cls, *pred_xywh = pred
            if pred_cls != gt_cls:
                continue
            v = iou(xywh_to_xyxy(*gt_xywh), xywh_to_xyxy(*pred_xywh))
            if v > best_iou:
                best_iou, best_pi = v, pi
        if best_iou >= iou_thresh:
            tp += 1
            matched_gt.add(gi)
            matched_pred.add(best_pi)
    fn = len(gt_boxes) - tp   # 원본에만 있는 박스
    fp = len(pred_boxes) - tp  # 자동에만 있는 박스
    return tp, fp, fn


def main():
    args = parse_args()
    auto_dir = Path(args.auto)
    src_dir  = Path(args.src)

    if not auto_dir.exists():
        print(f"[ERROR] 자동 라벨 폴더 없음: {auto_dir}"); return
    if not src_dir.exists():
        print(f"[ERROR] 원본 폴더 없음: {src_dir}"); return

    auto_files = {p.name for p in auto_dir.glob("*.txt")}
    src_files  = {p.name for p in src_dir.glob("*.txt")}

    only_auto   = sorted(auto_files - src_files)
    only_src    = sorted(src_files  - auto_files)
    common      = sorted(auto_files & src_files)

    # ── 1. 파일 수 비교 ──────────────────────────────────────────────────────
    print("=" * 60)
    print("[ 파일 개수 비교 ]")
    print(f"  자동 라벨  : {len(auto_files):>5}개  ({auto_dir})")
    print(f"  원본 라벨  : {len(src_files):>5}개  ({src_dir})")
    print(f"  공통 파일  : {len(common):>5}개")
    print(f"  자동에만   : {len(only_auto):>5}개")
    print(f"  원본에만   : {len(only_src):>5}개")

    if only_auto:
        print("\n  [자동에만 있는 파일 (처음 10개)]")
        for f in only_auto[:10]:
            print(f"    {f}")
    if only_src:
        print("\n  [원본에만 있는 파일 (처음 10개)]")
        for f in only_src[:10]:
            print(f"    {f}")

    # ── 2. 공통 파일 박스 수 / 클래스 분포 비교 ──────────────────────────────
    print("\n" + "=" * 60)
    print("[ 박스 수 비교 (공통 파일 기준) ]")

    total_gt_boxes = total_pred_boxes = 0
    auto_cls_counter = Counter()
    src_cls_counter  = Counter()
    total_tp = total_fp = total_fn = 0

    files_more_auto = []   # 자동이 더 많은 파일
    files_more_src  = []   # 원본이 더 많은 파일
    files_no_pred   = []   # 자동 박스가 0개인 파일

    for fname in common:
        gt   = load_labels(src_dir  / fname)
        pred = load_labels(auto_dir / fname)

        total_gt_boxes   += len(gt)
        total_pred_boxes += len(pred)

        for cls, *_ in gt:   src_cls_counter[cls]  += 1
        for cls, *_ in pred: auto_cls_counter[cls] += 1

        tp, fp, fn = match_boxes(gt, pred, args.iou)
        total_tp += tp; total_fp += fp; total_fn += fn

        diff = len(pred) - len(gt)
        if diff > 0:
            files_more_auto.append((fname, len(gt), len(pred)))
        elif diff < 0:
            files_more_src.append((fname, len(gt), len(pred)))
        if len(pred) == 0:
            files_no_pred.append(fname)

        if args.detail:
            mark = "✓" if len(gt) == len(pred) else "△"
            print(f"  {mark} {fname}  원본={len(gt)} 자동={len(pred)}")

    print(f"\n  공통 {len(common)}개 파일 합계")
    print(f"  원본 총 박스 수  : {total_gt_boxes:>6}")
    print(f"  자동 총 박스 수  : {total_pred_boxes:>6}")
    print(f"  차이 (자동-원본) : {total_pred_boxes - total_gt_boxes:>+6}")

    print(f"\n  자동 > 원본 (과검출) 파일 수 : {len(files_more_auto)}")
    print(f"  자동 < 원본 (미검출) 파일 수 : {len(files_more_src)}")
    print(f"  자동 박스 0개  파일 수       : {len(files_no_pred)}")

    if files_more_auto:
        print("\n  [과검출 상위 10개]")
        for f, g, p in sorted(files_more_auto, key=lambda x: x[2]-x[1], reverse=True)[:10]:
            print(f"    {f}  원본={g} 자동={p} (+{p-g})")
    if files_more_src:
        print("\n  [미검출 상위 10개]")
        for f, g, p in sorted(files_more_src, key=lambda x: x[1]-x[2], reverse=True)[:10]:
            print(f"    {f}  원본={g} 자동={p} (-{g-p})")

    # ── 3. IoU 기반 매칭 정밀도 ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"[ IoU ≥ {args.iou} 기반 박스 매칭 (공통 파일) ]")
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0
    recall    = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
    print(f"  TP (올바른 검출)  : {total_tp:>6}")
    print(f"  FP (과검출)       : {total_fp:>6}")
    print(f"  FN (미검출)       : {total_fn:>6}")
    print(f"  Precision         : {precision:.4f}")
    print(f"  Recall            : {recall:.4f}")
    print(f"  F1 Score          : {f1:.4f}")

    # ── 4. 클래스 분포 비교 ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("[ 클래스별 박스 수 비교 (공통 파일) ]")
    all_cls = sorted(set(src_cls_counter) | set(auto_cls_counter))
    print(f"  {'cls':>4}  {'원본':>7}  {'자동':>7}  {'차이':>7}")
    print(f"  {'-'*4}  {'-'*7}  {'-'*7}  {'-'*7}")
    for cls in all_cls:
        g = src_cls_counter[cls]
        p = auto_cls_counter[cls]
        print(f"  {cls:>4}  {g:>7}  {p:>7}  {p-g:>+7}")

    print("=" * 60)


if __name__ == "__main__":
    main()
