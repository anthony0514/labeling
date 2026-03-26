"""
라벨 파일 유효성 검사 스크립트
- 이미지와 라벨 쌍이 맞는지 확인
- 라벨 값 범위(0~1) 체크
- 클래스 ID 범위 체크
"""
import argparse
import yaml
from pathlib import Path


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(description="라벨 유효성 검사")
    parser.add_argument("--data", type=str, default="dataset.yaml",
                        help="dataset yaml 경로")
    parser.add_argument("--split", type=str, default="train", choices=["train", "val"],
                        help="검사할 split")
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.data) as f:
        cfg = yaml.safe_load(f)

    dataset_root = Path(cfg.get("path", "dataset"))
    nc = cfg.get("nc", None)
    img_dir = dataset_root / "images" / args.split
    label_dir = dataset_root / "labels" / args.split

    images = sorted([p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS])
    print(f"[{args.split}] 이미지 수: {len(images)}")

    errors = []
    for img_path in images:
        label_path = label_dir / (img_path.stem + ".txt")
        if not label_path.exists():
            errors.append(f"라벨 없음: {img_path.name}")
            continue

        with open(label_path) as f:
            lines = [l.strip() for l in f if l.strip()]

        for i, line in enumerate(lines, 1):
            parts = line.split()
            if len(parts) != 5:
                errors.append(f"{label_path.name} L{i}: 컬럼 수 오류 ({len(parts)}개)")
                continue
            cls, xc, yc, w, h = parts
            cls = int(cls)
            xc, yc, w, h = float(xc), float(yc), float(w), float(h)

            if nc and cls >= nc:
                errors.append(f"{label_path.name} L{i}: 클래스 ID {cls} >= nc({nc})")
            for val, name in [(xc, "x_center"), (yc, "y_center"), (w, "width"), (h, "height")]:
                if not (0.0 <= val <= 1.0):
                    errors.append(f"{label_path.name} L{i}: {name}={val} 범위 초과")

    if errors:
        print(f"\n오류 {len(errors)}건:")
        for e in errors:
            print(f"  [!] {e}")
    else:
        print("모든 라벨 정상.")


if __name__ == "__main__":
    main()
