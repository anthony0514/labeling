#!/usr/bin/env python3
"""YOLO txt 라벨의 클래스 번호를 새 정의에 맞춰 다시 매긴다.

기존 이름 목록(obj.names.bak)과 새 정의(classes.yaml)를 이름으로 대조해서
  - 번호가 바뀐 클래스는 새 번호로 치환
  - 없어진 클래스의 박스 줄은 삭제
기본은 미리보기(dry-run)이고, --apply 를 줘야 실제로 쓴다.
적용 전에는 대상 폴더의 txt 전체를 zip 으로 백업한다.

    python scripts/remap_labels.py <라벨폴더>                  # 미리보기
    python scripts/remap_labels.py <라벨폴더> --apply          # 적용
"""
import argparse
import sys
import zipfile
from collections import Counter
from datetime import datetime
from pathlib import Path

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent.parent


def load_old_names(path: Path) -> list:
    return [l.strip() for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def load_new_names(path: Path) -> list:
    d = yaml.safe_load(path.read_text(encoding="utf-8"))
    names = d["names"]
    if isinstance(names, dict):
        return [names[i] for i in range(d.get("nc", len(names)))]
    return list(names)


def build_mapping(old: list, new: list):
    """이름 기준 {기존번호: 새번호} 와 없어진 [(번호, 이름)] 반환."""
    new_idx = {n: i for i, n in enumerate(new)}
    mapping, removed = {}, []
    for i, n in enumerate(old):
        if n in new_idx:
            mapping[i] = new_idx[n]
        else:
            removed.append((i, n))
    return mapping, removed


def remap_file(txt: Path, mapping: dict):
    """반환: (새 내용, 바뀐 줄 수, 지운 줄 수, 매핑에 없는 번호 Counter)"""
    out, changed, dropped = [], 0, 0
    unknown = Counter()
    for line in txt.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) != 5:
            if line.strip():
                out.append(line)
            continue
        try:
            cls = int(parts[0])
        except ValueError:
            out.append(line)
            continue
        if cls not in mapping:
            unknown[cls] += 1
            dropped += 1
            continue
        if mapping[cls] != cls:
            parts[0] = str(mapping[cls])
            changed += 1
        out.append(" ".join(parts))
    return ("\n".join(out) + "\n" if out else ""), changed, dropped, unknown


def backup(folder: Path, txts: list) -> Path:
    stamp = datetime.now().strftime("%y%m%d_%H%M%S")
    zp = folder.parent / f"{folder.name}_labels_backup_{stamp}.zip"
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as zf:
        for t in txts:
            zf.write(t, str(t.relative_to(folder)))
    return zp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", type=Path, help="라벨 txt 가 있는 폴더 (하위 폴더 포함)")
    ap.add_argument("--config", type=Path,
                    default=SCRIPT_DIR / "config_defect_canada",
                    help="설정 폴더 (기본: config_defect_canada)")
    ap.add_argument("--old", type=Path, help="기존 이름 파일 (기본: <config>/obj.names.bak)")
    ap.add_argument("--new", type=Path, help="새 정의 yaml (기본: <config>/classes.yaml)")
    ap.add_argument("--apply", action="store_true", help="실제로 파일을 수정")
    args = ap.parse_args()

    old_path = args.old or (args.config / "obj.names.bak")
    new_path = args.new or (args.config / "classes.yaml")
    for p in (args.folder, old_path, new_path):
        if not p.exists():
            sys.exit(f"없는 경로: {p}")

    old, new = load_old_names(old_path), load_new_names(new_path)
    mapping, removed = build_mapping(old, new)

    print(f"기존 {len(old)}개 → 신규 {len(new)}개")
    print("없어진 클래스:",
          ", ".join(f"{i}번 {n}" for i, n in removed) if removed else "없음")
    moved = [(o, n) for o, n in sorted(mapping.items()) if o != n]
    print(f"번호 바뀜: {len(moved)}개", f"(예: {moved[:3]})" if moved else "")

    txts = sorted(args.folder.rglob("*.txt"))
    if not txts:
        sys.exit(f"txt 파일이 없습니다: {args.folder}")

    files_changed = lines_changed = lines_dropped = 0
    unknown_total = Counter()
    pending = []
    for t in txts:
        content, ch, dr, unk = remap_file(t, mapping)
        unknown_total += unk
        if ch or dr:
            files_changed += 1
            lines_changed += ch
            lines_dropped += dr
            pending.append((t, content))

    print(f"\n대상 txt {len(txts)}개 중 {files_changed}개 변경 예정")
    print(f"  번호 치환: {lines_changed}줄")
    print(f"  삭제(없어진 클래스): {lines_dropped}줄")
    if unknown_total:
        print("  삭제된 번호별:",
              ", ".join(f"{c}번 {n}줄" for c, n in sorted(unknown_total.items())))

    if not args.apply:
        print("\n미리보기입니다. 실제 적용하려면 --apply 를 붙이세요.")
        return

    zp = backup(args.folder, txts)
    print(f"\n백업: {zp}")
    for t, content in pending:
        t.write_text(content, encoding="utf-8")
    print(f"적용 완료: {files_changed}개 파일")


if __name__ == "__main__":
    main()
