"""
특정 클래스만 자동 라벨로 교체하는 스크립트

기존 수작업 라벨 파일에서 지정한 클래스 ID 줄만 제거하고,
자동 라벨링 결과의 해당 클래스 줄로 교체합니다.
나머지 클래스는 그대로 유지됩니다.

사용법:
  python scripts/overlay_classes.py \
    --auto_label_dir runs/detect/predict/labels \
    --target_label_dir dataset/labels/train \
    --classes 22 29 30
"""
import argparse
import shutil
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="특정 클래스만 자동 라벨로 교체")
    parser.add_argument("--auto_label_dir", type=str,
                        default="runs/detect/predict/labels",
                        help="자동 라벨링 결과 폴더 (predict labels)")
    parser.add_argument("--target_label_dir", type=str,
                        default="dataset/labels/train",
                        help="교체 대상 기존 라벨 폴더")
    parser.add_argument("--classes", type=int, nargs="+", default=[22, 29, 30],
                        help="교체할 클래스 ID 목록 (기본: 22 29 30)")
    parser.add_argument("--dry_run", action="store_true",
                        help="실제로 변경하지 않고 미리보기만")
    parser.add_argument("--backup", action="store_true",
                        help="기존 파일을 .bak 으로 백업 후 교체")
    return parser.parse_args()


def main():
    args = parse_args()

    auto_dir = Path(args.auto_label_dir)
    target_dir = Path(args.target_label_dir)
    replace_classes = set(str(c) for c in args.classes)

    if not auto_dir.exists():
        print(f"[ERROR] 자동 라벨 폴더 없음: {auto_dir}")
        return
    if not target_dir.exists():
        print(f"[ERROR] 대상 라벨 폴더 없음: {target_dir}")
        return

    print(f"교체 대상 클래스 : {sorted(args.classes)}")
    print(f"자동 라벨 소스   : {auto_dir}")
    print(f"교체 대상 폴더   : {target_dir}")
    print(f"{'[DRY RUN] ' if args.dry_run else ''}{'[백업 ON] ' if args.backup else ''}")
    print()

    auto_files = list(auto_dir.glob("*.txt"))

    stats = {"overlay": 0, "new": 0, "no_change": 0}

    for auto_path in sorted(auto_files):
        stem = auto_path.stem
        target_path = target_dir / (stem + ".txt")

        # 자동 라벨에서 교체 대상 클래스 줄만 추출
        auto_lines = [
            l.strip() for l in auto_path.read_text().splitlines()
            if l.strip() and l.strip().split()[0] in replace_classes
        ]

        if target_path.exists():
            # 기존 파일에서 교체 대상 클래스 줄 제거
            existing_lines = [
                l.strip() for l in target_path.read_text().splitlines()
                if l.strip() and l.strip().split()[0] not in replace_classes
            ]
            merged = existing_lines + auto_lines

            if not args.dry_run:
                if args.backup:
                    shutil.copy2(target_path, target_path.with_suffix(".bak"))
                target_path.write_text("\n".join(merged) + "\n" if merged else "")

            print(f"  [OVERLAY] {stem}.txt  "
                  f"기존 {len(existing_lines)}줄 유지 + 자동 {len(auto_lines)}줄 교체")
            stats["overlay"] += 1

        else:
            # 기존 파일 없으면 새로 생성
            if not args.dry_run:
                target_path.write_text("\n".join(auto_lines) + "\n" if auto_lines else "")
            print(f"  [NEW]     {stem}.txt  자동 라벨 {len(auto_lines)}줄 신규 생성")
            stats["new"] += 1

    print()
    print(f"완료{'(dry run)' if args.dry_run else ''}")
    print(f"  교체(overlay) : {stats['overlay']}개")
    print(f"  신규 생성     : {stats['new']}개")


if __name__ == "__main__":
    main()
