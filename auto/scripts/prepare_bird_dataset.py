"""
prepare_bird_dataset.py
========================
source/bird_labeled_1 + bird_labeled_2 를 bird 전용 데이터셋으로 구성.

현황:
  - class 0: bird    (~7454 boxes)
  - class 1: drone   (없음 - 추후 추가용으로 자리 보존)
  - class 2: uncertain (~11 boxes)

결과:
  auto/dataset_bird/
    images/train/   images/val/
    labels/train/   labels/val/
  auto/dataset_bird.yaml

사용법:
    cd ~/labeling/auto
    source venv/bin/activate
    python scripts/prepare_bird_dataset.py [--dry-run] [--val-ratio 0.1]
"""

import argparse
import random
import shutil
from pathlib import Path

SCRIPT_DIR  = Path(__file__).resolve().parent
AUTO_DIR    = SCRIPT_DIR.parent
SOURCE_DIR  = AUTO_DIR.parent / "source"

BIRD_SOURCES = ["bird_labeled_1", "bird_labeled_2"]

DATASET_DIR  = AUTO_DIR / "dataset_bird"
TRAIN_IMG    = DATASET_DIR / "images" / "train"
VAL_IMG      = DATASET_DIR / "images" / "val"
TRAIN_LBL    = DATASET_DIR / "labels" / "train"
VAL_LBL      = DATASET_DIR / "labels" / "val"
YAML_PATH    = AUTO_DIR / "dataset_bird.yaml"

IMG_EXTS  = {".jpg", ".jpeg", ".png", ".bmp"}
VAL_RATIO = 0.1
SEED      = 42


def collect_pairs():
    pairs = []
    for name in BIRD_SOURCES:
        folder = SOURCE_DIR / name
        if not folder.exists():
            print(f"  [경고] 폴더 없음: {folder}")
            continue
        imgs = {p.stem: p for p in folder.iterdir()
                if p.suffix.lower() in IMG_EXTS}
        txts = {p.stem: p for p in folder.glob("*.txt")}
        matched = [(imgs[s], txts[s]) for s in imgs if s in txts]
        print(f"  {name}: {len(matched)}쌍")
        pairs.extend(matched)
    return pairs


def write_yaml():
    content = (
        f"path: {DATASET_DIR}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"\n"
        f"nc: 3\n"
        f"names:\n"
        f"  0: bird\n"
        f"  1: drone\n"
        f"  2: uncertain\n"
    )
    YAML_PATH.write_text(content)
    print(f"  yaml → {YAML_PATH}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run",   action="store_true")
    parser.add_argument("--val-ratio", type=float, default=VAL_RATIO)
    parser.add_argument("--seed",      type=int,   default=SEED)
    args = parser.parse_args()

    dry = args.dry_run
    random.seed(args.seed)

    if dry:
        print("[DRY-RUN] 실제 파일 변경 없음\n")

    print("소스 수집:")
    pairs = collect_pairs()
    if not pairs:
        print("데이터 없음. 종료.")
        return

    print(f"\n총 {len(pairs)}쌍")

    # 클래스 분포 출력
    class_cnt = {}
    for _, txt in pairs:
        for line in txt.read_text().strip().splitlines():
            if line.strip():
                c = int(line.split()[0])
                class_cnt[c] = class_cnt.get(c, 0) + 1
    names = {0: "bird", 1: "drone", 2: "uncertain"}
    print("클래스 분포:")
    for c, cnt in sorted(class_cnt.items()):
        print(f"  [{c}] {names.get(c, '?'):12s}: {cnt:5d} boxes")

    # 이미 존재하는 파일 확인
    existing = set()
    if TRAIN_IMG.exists():
        existing |= {p.stem for p in TRAIN_IMG.iterdir()}
    if VAL_IMG.exists():
        existing |= {p.stem for p in VAL_IMG.iterdir()}
    if existing:
        print(f"\n기존 데이터셋 {len(existing)}개 감지 → 덮어쓰기")

    # train / val 분할
    random.shuffle(pairs)
    n_val   = max(1, int(len(pairs) * args.val_ratio))
    n_train = len(pairs) - n_val
    val_pairs   = pairs[:n_val]
    train_pairs = pairs[n_val:]
    print(f"\n분할: train {n_train}, val {n_val}")

    if not dry:
        for d in [TRAIN_IMG, VAL_IMG, TRAIN_LBL, VAL_LBL]:
            d.mkdir(parents=True, exist_ok=True)

    def copy(img, txt, img_dir, lbl_dir):
        if not dry:
            shutil.copy2(img, img_dir / img.name)
            shutil.copy2(txt, lbl_dir / (img.stem + ".txt"))

    for img, txt in train_pairs:
        copy(img, txt, TRAIN_IMG, TRAIN_LBL)
    for img, txt in val_pairs:
        copy(img, txt, VAL_IMG, VAL_LBL)

    if not dry:
        write_yaml()

    print(f"\n{'[DRY] ' if dry else ''}완료!")
    print(f"  dataset_bird/images/train : {n_train}장")
    print(f"  dataset_bird/images/val   : {n_val}장")
    if not dry:
        print(f"  yaml : {YAML_PATH}")
        print(f"\n다음 단계:")
        print(f"  python scripts/train_bird.py")


if __name__ == "__main__":
    main()
