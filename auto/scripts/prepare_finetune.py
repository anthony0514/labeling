"""
prepare_finetune.py
====================
파인튜닝용 데이터셋 준비 스크립트.

1. 기존 dataset/images/train+val 유지
2. source/ 아래 검수 완료 폴더에서 이미지+txt 쌍 수집
   - 이미지가 없는 경우 auto/unlabeled 에서 찾아서 복사
3. train:val = 9:1 로 분할해 dataset에 추가
4. dataset.yaml 업데이트

사용법:
    cd ~/labeling/auto
    source venv/bin/activate
    python scripts/prepare_finetune.py [--dry-run]
"""

import argparse
import random
import shutil
from pathlib import Path

# ── 경로 설정 ──────────────────────────────────────────────────────────────
SCRIPT_DIR   = Path(__file__).resolve().parent
AUTO_DIR     = SCRIPT_DIR.parent
SOURCE_DIR   = AUTO_DIR.parent / "source"
UNLABELED    = AUTO_DIR / "unlabeled"
DATASET_DIR  = AUTO_DIR / "dataset"
TRAIN_IMG    = DATASET_DIR / "images" / "train"
VAL_IMG      = DATASET_DIR / "images" / "val"
TRAIN_LBL    = DATASET_DIR / "labels" / "train"
VAL_LBL      = DATASET_DIR / "labels" / "val"

# 수집 대상: source 아래 폴더명에 "검수완료" 또는 "labeled" 포함
INCLUDE_KEYWORDS = ["검수완료", "labeled", "autolabeled"]

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
VAL_RATIO = 0.1
SEED      = 42


def find_pairs(folder: Path, unlabeled_fallback: Path):
    """(img_path, txt_path) 쌍 목록 반환. 이미지가 없으면 unlabeled에서 찾음."""
    pairs = []
    txts  = list(folder.glob("*.txt"))
    for txt in txts:
        stem = txt.stem
        # 같은 폴더에서 이미지 찾기
        img = None
        for ext in IMG_EXTS:
            candidate = folder / (stem + ext)
            if candidate.exists():
                img = candidate
                break
        # 없으면 unlabeled 에서 찾기
        if img is None and unlabeled_fallback.exists():
            for ext in IMG_EXTS:
                candidate = unlabeled_fallback / (stem + ext)
                if candidate.exists():
                    img = candidate
                    break
        if img is None:
            continue
        pairs.append((img, txt))
    return pairs


def copy_pair(img: Path, txt: Path, img_dst_dir: Path, lbl_dst_dir: Path, dry_run: bool):
    img_dst = img_dst_dir / img.name
    lbl_dst = lbl_dst_dir / (img.stem + ".txt")
    if img_dst.exists():
        return False   # 이미 존재
    if not dry_run:
        shutil.copy2(img, img_dst)
        shutil.copy2(txt, lbl_dst)
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="실제 복사 없이 결과만 출력")
    parser.add_argument("--val-ratio", type=float, default=VAL_RATIO)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    dry = args.dry_run
    random.seed(args.seed)

    if dry:
        print("[DRY-RUN 모드] 실제 파일 변경 없음\n")

    # ── 대상 폴더 수집 ────────────────────────────────────────────────────
    target_folders = [
        d for d in SOURCE_DIR.iterdir()
        if d.is_dir() and any(kw in d.name for kw in INCLUDE_KEYWORDS)
    ]
    print(f"대상 폴더 {len(target_folders)}개:")
    for f in sorted(target_folders):
        print(f"  {f.name}")
    print()

    # ── 기존 train/val 파일명 수집 (중복 방지) ────────────────────────────
    existing = set()
    for p in list(TRAIN_IMG.glob("*")) + list(VAL_IMG.glob("*")):
        existing.add(p.stem)
    print(f"기존 데이터셋: {len(existing)}개 (중복 스킵)\n")

    # ── 신규 pairs 수집 ───────────────────────────────────────────────────
    all_pairs = []
    for folder in sorted(target_folders):
        pairs = find_pairs(folder, UNLABELED)
        new_pairs = [(img, txt) for img, txt in pairs if img.stem not in existing]
        print(f"  {folder.name}: 총 {len(pairs)}쌍, 신규 {len(new_pairs)}쌍")
        all_pairs.extend(new_pairs)

    print(f"\n신규 추가 가능: {len(all_pairs)}쌍")
    if not all_pairs:
        print("추가할 데이터 없음.")
        return

    # ── train / val 분할 ─────────────────────────────────────────────────
    random.shuffle(all_pairs)
    n_val   = max(1, int(len(all_pairs) * args.val_ratio))
    n_train = len(all_pairs) - n_val
    val_pairs   = all_pairs[:n_val]
    train_pairs = all_pairs[n_val:]
    print(f"분할: train {n_train}개, val {n_val}개\n")

    if not dry:
        for d in [TRAIN_IMG, VAL_IMG, TRAIN_LBL, VAL_LBL]:
            d.mkdir(parents=True, exist_ok=True)

    # ── 복사 ─────────────────────────────────────────────────────────────
    added_train = sum(copy_pair(img, txt, TRAIN_IMG, TRAIN_LBL, dry)
                      for img, txt in train_pairs)
    added_val   = sum(copy_pair(img, txt, VAL_IMG,   VAL_LBL,   dry)
                      for img, txt in val_pairs)

    print(f"{'[DRY] ' if dry else ''}복사 완료: train +{added_train}, val +{added_val}")
    print(f"최종 train: {len(list(TRAIN_IMG.glob('*')))}장, "
          f"val: {len(list(VAL_IMG.glob('*')))}장")


if __name__ == "__main__":
    main()
