"""
특정 클래스만 추려서 새 서브 데이터셋을 생성하는 스크립트

- 원본 데이터셋은 절대 수정하지 않음
- 지정한 클래스가 1개 이상 포함된 이미지만 복사
- 클래스 ID를 0부터 재매핑
- 새 dataset yaml 자동 생성

사용법:
  python scripts/filter_classes.py --classes 22 29 30
"""

import argparse
import shutil
import yaml
from pathlib import Path


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(description="클래스 필터링 서브 데이터셋 생성")
    parser.add_argument("--classes", type=int, nargs="+", required=True,
                        help="추출할 원본 클래스 ID 목록 (예: 22 29 30)")
    parser.add_argument("--src_yaml", type=str, default="dataset.yaml",
                        help="원본 dataset yaml 경로")
    parser.add_argument("--out_dir", type=str, default=None,
                        help="출력 데이터셋 루트 (기본: dataset_cls22_29_30 형식)")
    parser.add_argument("--out_yaml", type=str, default=None,
                        help="출력 yaml 파일명 (기본: dataset_cls22_29_30.yaml 형식)")
    return parser.parse_args()


def sep(title=""):
    width = 60
    if title:
        pad = (width - len(title) - 2) // 2
        print(f"\n{'─'*pad} {title} {'─'*pad}")
    else:
        print("─" * width)


def main():
    args = parse_args()
    target_classes = sorted(args.classes)
    cls_tag = "_".join(str(c) for c in target_classes)

    # ── 경로 설정 ─────────────────────────────────────────────
    out_dir  = Path(args.out_dir  or f"dataset_cls{cls_tag}")
    out_yaml = Path(args.out_yaml or f"dataset_cls{cls_tag}.yaml")

    # ── 원본 yaml 로드 ────────────────────────────────────────
    src_yaml = Path(args.src_yaml)
    if not src_yaml.exists():
        print(f"[ERROR] dataset yaml 없음: {src_yaml}")
        return

    with open(src_yaml) as f:
        cfg = yaml.safe_load(f)

    src_root  = Path(cfg.get("path", "dataset"))
    all_names = cfg.get("names", {})

    # 원본 클래스명 확인
    sep("필터링 설정")
    remap = {orig: new for new, orig in enumerate(target_classes)}
    print(f"  추출 클래스:")
    for orig_id, new_id in remap.items():
        name = all_names.get(orig_id, f"class{orig_id}")
        print(f"    원본 class {orig_id:2d} ({name})  →  새 class {new_id}")
    print(f"\n  원본 데이터셋 : {src_root}")
    print(f"  출력 데이터셋 : {out_dir}")
    print(f"  출력 yaml     : {out_yaml}")

    # ── split 별 처리 ─────────────────────────────────────────
    total_copied = {"train": 0, "val": 0}
    total_skipped = {"train": 0, "val": 0}

    for split in ["train", "val"]:
        sep(f"{split} 처리")

        img_src = src_root / "images" / split
        lbl_src = src_root / "labels" / split
        img_dst = out_dir  / "images" / split
        lbl_dst = out_dir  / "labels" / split

        if not img_src.exists() or not lbl_src.exists():
            print(f"  [SKIP] {split} 폴더 없음")
            continue

        img_dst.mkdir(parents=True, exist_ok=True)
        lbl_dst.mkdir(parents=True, exist_ok=True)

        label_files = sorted(lbl_src.glob("*.txt"))
        copied = skipped = 0

        for lbl_path in label_files:
            lines = [l.strip() for l in lbl_path.read_text().splitlines() if l.strip()]

            # 해당 클래스 라인만 필터링
            filtered = [l for l in lines if int(l.split()[0]) in remap]
            if not filtered:
                skipped += 1
                continue

            # 이미지 파일 탐색
            img_path = None
            for ext in IMG_EXTS:
                candidate = img_src / (lbl_path.stem + ext)
                if candidate.exists():
                    img_path = candidate
                    break
            if img_path is None:
                print(f"  [WARN] 이미지 없음: {lbl_path.stem}")
                skipped += 1
                continue

            # 클래스 ID 재매핑
            new_lines = []
            for line in filtered:
                parts = line.split()
                parts[0] = str(remap[int(parts[0])])
                new_lines.append(" ".join(parts))

            # 복사
            shutil.copy2(img_path, img_dst / img_path.name)
            (lbl_dst / lbl_path.name).write_text("\n".join(new_lines) + "\n")
            copied += 1

        total_copied[split]  = copied
        total_skipped[split] = skipped
        print(f"  복사: {copied}쌍  /  스킵(해당 클래스 없음): {skipped}개")

    # ── 새 yaml 생성 ──────────────────────────────────────────
    new_names = {new_id: all_names.get(orig_id, f"class{orig_id}")
                 for orig_id, new_id in remap.items()}
    new_cfg = {
        "path": str(out_dir),
        "train": "images/train",
        "val": "images/val",
        "nc": len(target_classes),
        "names": new_names,
    }
    with open(out_yaml, "w") as f:
        yaml.dump(new_cfg, f, allow_unicode=True, sort_keys=False)

    # ── 최종 요약 ─────────────────────────────────────────────
    sep("완료 요약")
    print(f"  train  {total_copied['train']}쌍  /  val  {total_copied['val']}쌍  복사됨")
    print(f"  새 yaml: {out_yaml}")
    print(f"\n  다음 단계 — 학습:")
    print(f"    python scripts/train.py --data {out_yaml} --name train_cls{cls_tag}")
    sep()


if __name__ == "__main__":
    main()
