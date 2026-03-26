"""
검수 완료된 자동 라벨 데이터를 기존 dataset에 병합하는 스크립트

사용법:
  python scripts/merge_dataset.py \
    --img_src runs/detect/predict \
    --label_src runs/detect/predict/labels \
    --split train   # train 또는 val
"""
import argparse
import shutil
from pathlib import Path


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(description="검수 완료 라벨 데이터 병합")
    parser.add_argument("--img_src", type=str, required=True,
                        help="추가할 이미지 폴더 경로")
    parser.add_argument("--label_src", type=str, required=True,
                        help="추가할 라벨(.txt) 폴더 경로")
    parser.add_argument("--split", type=str, default="train", choices=["train", "val"],
                        help="병합할 split (train / val)")
    parser.add_argument("--dataset_root", type=str, default="dataset",
                        help="dataset 루트 경로")
    parser.add_argument("--dry_run", action="store_true",
                        help="실제로 복사하지 않고 결과만 미리보기")
    return parser.parse_args()


def main():
    args = parse_args()

    img_src = Path(args.img_src)
    label_src = Path(args.label_src)
    img_dst = Path(args.dataset_root) / "images" / args.split
    label_dst = Path(args.dataset_root) / "labels" / args.split

    if not img_src.exists():
        print(f"[ERROR] 이미지 소스 경로 없음: {img_src}")
        return
    if not label_src.exists():
        print(f"[ERROR] 라벨 소스 경로 없음: {label_src}")
        return

    img_dst.mkdir(parents=True, exist_ok=True)
    label_dst.mkdir(parents=True, exist_ok=True)

    images = [p for p in img_src.iterdir() if p.suffix.lower() in IMG_EXTS]
    copied = 0
    skipped = 0

    for img_path in sorted(images):
        label_path = label_src / (img_path.stem + ".txt")
        if not label_path.exists():
            print(f"  [SKIP] 라벨 없음: {img_path.name}")
            skipped += 1
            continue

        if not args.dry_run:
            shutil.copy2(img_path, img_dst / img_path.name)
            shutil.copy2(label_path, label_dst / label_path.name)
        else:
            print(f"  [DRY] {img_path.name} + {label_path.name}")
        copied += 1

    print(f"\n병합 완료 ({'dry run' if args.dry_run else '실제 복사'})")
    print(f"  복사: {copied}쌍  /  라벨 없어 스킵: {skipped}개")
    print(f"  대상: {img_dst}  /  {label_dst}")


if __name__ == "__main__":
    main()
