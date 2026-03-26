"""
Cleanlab ObjectLab 기반 라벨 품질 검수 스크립트

흐름:
  1. best.pt 로 source 이미지 추론 (confidence 포함 예측값 캐싱)
  2. 원본 GT 라벨 로드 → Cleanlab 형식으로 변환
  3. find_label_issues 실행 → 의심 라벨 점수화
  4. 의심 이미지 시각화 + CSV 결과 저장 (review/ 폴더)

사용법:
  python scripts/cleanlab_review.py
  python scripts/cleanlab_review.py --top 50        # 상위 50개만 시각화
  python scripts/cleanlab_review.py --no-cache      # 추론 재실행
  python scripts/cleanlab_review.py --conf 0.05     # 예측 conf 임계값 조정
"""
import argparse
import pickle
import csv
import shutil
from pathlib import Path
import numpy as np
import cv2

PROJECT_ROOT  = Path(__file__).resolve().parent.parent
SOURCE_DIR    = Path.home() / "labeling/source/안재현_라벨링_500_5"
MODEL_PATH    = PROJECT_ROOT / "runs/detect/train/weights/best.pt"
REVIEW_DIR    = Path.home() / "labeling/review"
CACHE_FILE    = PROJECT_ROOT / "scripts/.pred_cache.pkl"

COLORS = [
    (255, 56,  56),  (255, 157,  151), (255, 212,   0), (0,   255,  76),
    (0,   152, 255), (132,  56, 255),  (255,  77, 195), (255, 178,  29),
    (207, 210,  49), (72,  249, 10),   (146, 204, 23),  (61,  219, 134),
    (26,  147, 52),  (0,   212, 187),  (44,  153, 168), (0,   194, 255),
    (52,   69, 147), (100,  115, 255), (0,    24, 236), (132,  56, 255),
    (200,  82, 175), (252,  69, 133),  (219, 137, 148), (255, 109,  23),
    (0,    55, 109), (128, 0,     0),  (0,   128,   0), (0,     0, 128),
    (128, 128,   0), (128,  0,  128),  (0,   128, 128),
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",    default=str(MODEL_PATH))
    p.add_argument("--src",      default=str(SOURCE_DIR))
    p.add_argument("--img-dir",  default=str(PROJECT_ROOT / "unlabeled"),
                   help="추론에 사용할 이미지 폴더")
    p.add_argument("--review",   default=str(REVIEW_DIR))
    p.add_argument("--conf",     type=float, default=0.01,
                   help="YOLO 예측 confidence 임계값 (기본 0.01 — 낮을수록 더 많은 후보)")
    p.add_argument("--top",      type=int, default=100,
                   help="시각화할 의심 이미지 수")
    p.add_argument("--no-cache", action="store_true",
                   help="캐시 무시하고 추론 재실행")
    return p.parse_args()


# ── YOLO label (cx cy w h) → xyxy normalized ─────────────────────────────────
def yolo_to_xyxy(cx, cy, w, h):
    return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]


# ── GT 로드 ───────────────────────────────────────────────────────────────────
def load_gt(src_dir: Path, stem_list: list):
    labels = []
    for stem in stem_list:
        txt = src_dir / f"{stem}.txt"
        bboxes, cls_ids = [], []
        if txt.exists():
            for line in txt.read_text().strip().splitlines():
                parts = line.split()
                if len(parts) == 5:
                    c, cx, cy, w, h = int(parts[0]), *map(float, parts[1:])
                    bboxes.append(yolo_to_xyxy(cx, cy, w, h))
                    cls_ids.append(c)
        labels.append({
            "bboxes": np.array(bboxes,  dtype=float).reshape(-1, 4),
            "labels": np.array(cls_ids, dtype=int),
        })
    return labels


# ── YOLO 추론 → raw 결과 캐싱 (xyxy_norm + conf + cls) ───────────────────────
def run_inference(model_path: str, img_paths: list, conf: float, cache: Path, no_cache: bool):
    if not no_cache and cache.exists():
        print(f"[캐시] 예측 결과 로드: {cache}")
        with open(cache, "rb") as f:
            return pickle.load(f)

    from ultralytics import YOLO
    model = YOLO(model_path)
    print(f"[추론] {len(img_paths)}장 이미지 → conf={conf}  (잠시 기다려주세요...)")

    raw_preds = []   # 각 이미지: np.ndarray (N, 6) — [x1 y1 x2 y2 conf cls]
    batch_size = 32
    for i in range(0, len(img_paths), batch_size):
        batch = [str(p) for p in img_paths[i:i + batch_size]]
        results = model.predict(batch, conf=conf, verbose=False)
        for r in results:
            boxes = r.boxes
            if boxes is None or len(boxes) == 0:
                raw_preds.append(np.zeros((0, 6), dtype=float))
                continue
            xyxy  = boxes.xyxyn.cpu().numpy()
            confs = boxes.conf.cpu().numpy().reshape(-1, 1)
            cls   = boxes.cls.cpu().numpy().reshape(-1, 1)
            raw_preds.append(np.hstack([xyxy, confs, cls]))
        if (i // batch_size + 1) % 10 == 0:
            print(f"  {min(i + batch_size, len(img_paths))} / {len(img_paths)}")

    print(f"[추론 완료] 캐시 저장: {cache}")
    cache.parent.mkdir(parents=True, exist_ok=True)
    with open(cache, "wb") as f:
        pickle.dump(raw_preds, f)
    return raw_preds


# ── raw 예측 → Cleanlab 형식 변환 ─────────────────────────────────────────────
# predictions[i][k] = np.ndarray (M, 5) — [x1 y1 x2 y2 pred_prob] for class k
def to_cleanlab_preds(raw_preds: list, num_classes: int):
    cl_preds = []
    for raw in raw_preds:
        per_class = []
        for k in range(num_classes):
            if len(raw) == 0:
                per_class.append(np.zeros((0, 5), dtype=float))
            else:
                mask = raw[:, 5].astype(int) == k
                if mask.any():
                    per_class.append(raw[mask][:, :5])   # [x1 y1 x2 y2 conf]
                else:
                    per_class.append(np.zeros((0, 5), dtype=float))
        cl_preds.append(per_class)
    return cl_preds


# ── 이미지 위에 bbox 그리기 ───────────────────────────────────────────────────
def draw_boxes(img, boxes_xyxy_norm, cls_ids, scores=None, gt=True):
    h, w = img.shape[:2]
    for i, (box, cls) in enumerate(zip(boxes_xyxy_norm, cls_ids)):
        x1, y1, x2, y2 = (int(v * d) for v, d in zip(box, [w, h, w, h]))
        color = COLORS[int(cls) % len(COLORS)]
        lw    = 2 if gt else 1
        cv2.rectangle(img, (x1, y1), (x2, y2), color, lw)
        label = f"{'GT' if gt else 'PR'} {int(cls)}"
        if scores is not None:
            label += f" {scores[i]:.2f}"
        cv2.putText(img, label, (x1, max(y1 - 4, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    return img


# ── 시각화 저장 ───────────────────────────────────────────────────────────────
def save_visualization(stem: str, score: float, rank: int,
                       img_dir: Path, gt: dict, pred: np.ndarray,
                       out_dir: Path, class_names: dict):
    img_path = img_dir / f"{stem}.jpg"
    if not img_path.exists():
        img_path = next(img_dir.glob(f"{stem}.*"), None)
    if img_path is None:
        return

    img = cv2.imread(str(img_path))
    if img is None:
        return

    # GT (실선)
    if len(gt["bboxes"]):
        draw_boxes(img, gt["bboxes"], gt["labels"], gt=True)

    # Prediction (점선 느낌 — 얇게)
    if len(pred):
        draw_boxes(img, pred[:, :4], pred[:, 5].astype(int),
                   scores=pred[:, 4], gt=False)

    # 점수 헤더
    header = f"Rank#{rank}  score={score:.4f}  {stem}"
    cv2.putText(img, header, (6, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

    out_path = out_dir / f"rank{rank:04d}_{stem}.jpg"
    cv2.imwrite(str(out_path), img)


def main():
    args   = parse_args()
    src_dir   = Path(args.src)
    img_dir   = Path(args.img_dir)
    review_dir = Path(args.review)

    # 출력 폴더
    vis_dir = review_dir / "flagged_images"
    vis_dir.mkdir(parents=True, exist_ok=True)

    # 클래스 이름 (dataset.yaml)
    import yaml
    yaml_path = PROJECT_ROOT / "dataset.yaml"
    class_names = {}
    if yaml_path.exists():
        with open(yaml_path) as f:
            d = yaml.safe_load(f)
            class_names = d.get("names", {})

    # ── 이미지 목록 (GT 있는 것만) ───────────────────────────────────────────
    gt_txts  = sorted(src_dir.glob("*.txt"))
    stems    = [p.stem for p in gt_txts]
    img_paths = []
    valid_stems = []
    for stem in stems:
        p = img_dir / f"{stem}.jpg"
        if not p.exists():
            p = next(img_dir.glob(f"{stem}.*"), None)
        if p:
            img_paths.append(p)
            valid_stems.append(stem)

    print(f"GT 파일: {len(stems)}개 / 이미지 매칭: {len(valid_stems)}개")

    # ── GT 로드 ───────────────────────────────────────────────────────────────
    gt_labels = load_gt(src_dir, valid_stems)

    # ── 추론 ─────────────────────────────────────────────────────────────────
    num_classes = len(class_names) if class_names else 31
    raw_preds = run_inference(args.model, img_paths, args.conf,
                              CACHE_FILE, args.no_cache)
    raw_preds = raw_preds[:len(valid_stems)]
    preds = to_cleanlab_preds(raw_preds, num_classes)

    # ── Cleanlab 실행 ─────────────────────────────────────────────────────────
    print("\n[Cleanlab] 라벨 품질 점수 계산 중...")
    from cleanlab.object_detection.filter import find_label_issues
    from cleanlab.object_detection.rank   import get_label_quality_scores

    scores  = get_label_quality_scores(gt_labels, preds)
    issues  = find_label_issues(gt_labels, preds)

    n_issues = issues.sum()
    print(f"\n[결과] 의심 이미지: {n_issues}개 / 전체: {len(valid_stems)}개 "
          f"({n_issues / len(valid_stems) * 100:.1f}%)")

    # ── 점수 낮은 순 정렬 ─────────────────────────────────────────────────────
    ranked = sorted(enumerate(scores), key=lambda x: x[1])

    # ── CSV 저장 ──────────────────────────────────────────────────────────────
    csv_path = review_dir / "label_quality_scores.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["rank", "filename", "quality_score", "is_issue",
                         "gt_box_count", "pred_box_count"])
        for rank, (idx, score) in enumerate(ranked, 1):
            stem = valid_stems[idx]
            writer.writerow([
                rank, stem, f"{score:.6f}", int(issues[idx]),
                len(gt_labels[idx]["labels"]),
                len(preds[idx]) if len(preds[idx]) else 0,
            ])
    print(f"[저장] CSV: {csv_path}")

    # ── 시각화 (상위 N개) ─────────────────────────────────────────────────────
    top_n = min(args.top, len(ranked))
    print(f"[시각화] 품질 점수 하위 {top_n}개 이미지 저장 중...")

    # 기존 파일 정리
    for f in vis_dir.glob("*.jpg"):
        f.unlink()

    for rank, (idx, score) in enumerate(ranked[:top_n], 1):
        save_visualization(
            stem       = valid_stems[idx],
            score      = score,
            rank       = rank,
            img_dir    = img_dir,
            gt         = gt_labels[idx],
            pred       = raw_preds[idx],
            out_dir    = vis_dir,
            class_names= class_names,
        )
        if rank % 20 == 0:
            print(f"  {rank} / {top_n}")

    print(f"[저장] 시각화: {vis_dir}  ({top_n}장)")

    # ── 요약 출력 ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("[ 요약 ]")
    print(f"  검수 대상   : {len(valid_stems):>5}개 이미지")
    print(f"  의심 라벨   : {n_issues:>5}개  ({n_issues/len(valid_stems)*100:.1f}%)")
    print(f"  품질 점수 평균  : {scores.mean():.4f}")
    print(f"  품질 점수 최저  : {scores.min():.4f}  ({valid_stems[ranked[0][0]]})")
    print(f"  결과 CSV    : {csv_path}")
    print(f"  시각화 폴더 : {vis_dir}")
    print("=" * 55)


if __name__ == "__main__":
    main()
