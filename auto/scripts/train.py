"""
YOLOv8 모델 학습 스크립트
"""
import argparse
import sys
import time
import platform
from datetime import datetime
from pathlib import Path

import torch
import yaml


def get_device() -> str:
    """환경에 따라 최적 디바이스 자동 감지 (mps → cuda → cpu)."""
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"
from ultralytics import YOLO


# ── 색상 코드 ────────────────────────────────────────────────
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

def ok(msg):   print(f"{GREEN}  ✔  {msg}{RESET}")
def info(msg): print(f"{CYAN}  ℹ  {msg}{RESET}")
def warn(msg): print(f"{YELLOW}  ⚠  {msg}{RESET}")
def err(msg):  print(f"{RED}  ✘  {msg}{RESET}")
def sep(title=""):
    width = 60
    if title:
        pad = (width - len(title) - 2) // 2
        print(f"\n{BOLD}{'─'*pad} {title} {'─'*pad}{RESET}")
    else:
        print(f"{BOLD}{'─'*width}{RESET}")


# 프로젝트 루트 = scripts/ 의 상위 폴더 (어디서 실행해도 항상 고정)
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def parse_args():
    parser = argparse.ArgumentParser(description="YOLOv8 학습")
    parser.add_argument("--model", type=str, default="yolov8n.pt",
                        help="기본 모델 (yolov8n/s/m/l/x.pt)")
    parser.add_argument("--data", type=str, default="dataset.yaml",
                        help="dataset yaml 경로")
    parser.add_argument("--epochs", type=int, default=100,
                        help="학습 에폭 수")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="입력 이미지 크기")
    parser.add_argument("--batch", type=int, default=16,
                        help="배치 사이즈 (-1 = auto)")
    parser.add_argument("--device", type=str, default=get_device(),
                        help="학습 디바이스 (자동 감지 / mps / cuda / cpu / 0)")
    parser.add_argument("--name", type=str, default="train",
                        help="결과 저장 폴더 이름 (runs/detect/<name>)")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="val 시 confidence threshold — 낮을수록 NMS 부하 증가 (기본: 0.25)")
    parser.add_argument("--max_det", type=int, default=300,
                        help="이미지당 최대 검출 bbox 수 — 초과 시 NMS 타임아웃 발생 (기본: 300)")
    return parser.parse_args()


# ── 사전 검사 ─────────────────────────────────────────────────
def check_environment(args):
    sep("환경 검사")

    # Python / PyTorch 버전
    info(f"Python  : {sys.version.split()[0]}")
    info(f"PyTorch : {torch.__version__}")
    info(f"OS      : {platform.system()} {platform.machine()}")

    # 디바이스
    d = args.device
    if d == "mps":
        ok("MPS (Apple Silicon GPU) 사용") if torch.backends.mps.is_available() else warn("MPS 사용 불가 → CPU 전환")
    elif d == "cpu":
        warn("CPU 모드로 실행 (느림)")
    elif torch.cuda.is_available():
        ok(f"CUDA 사용 가능: {torch.cuda.get_device_name(0)}")
    else:
        warn(f"디바이스 '{d}' — CUDA 없음, CPU로 전환될 수 있음")

    print()


def check_model(args):
    sep("모델 검사")
    model_path = Path(args.model)

    if model_path.exists():
        size_mb = model_path.stat().st_size / 1_000_000
        ok(f"모델 파일 존재: {args.model}  ({size_mb:.1f} MB)")
    else:
        # pretrained 다운로드 대상인지 확인
        pretrained = {"yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt"}
        if args.model in pretrained:
            info(f"'{args.model}' 는 pretrained 모델 → 학습 시작 시 자동 다운로드됨")
        else:
            err(f"모델 파일을 찾을 수 없음: {args.model}")
            sys.exit(1)
    print()


def check_dataset(args):
    sep("데이터셋 검사")

    yaml_path = Path(args.data)
    if not yaml_path.exists():
        err(f"dataset yaml 없음: {args.data}")
        sys.exit(1)
    ok(f"dataset yaml 존재: {args.data}")

    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)

    dataset_root = Path(cfg.get("path", "."))
    nc = cfg.get("nc", "?")
    names = cfg.get("names", {})
    info(f"dataset 경로 : {dataset_root}")
    info(f"클래스 수    : {nc}")
    info(f"클래스 목록  : {list(names.values()) if isinstance(names, dict) else names}")

    IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    issues = []

    for split in ["train", "val"]:
        img_dir_key = cfg.get(split, f"images/{split}")
        img_dir = dataset_root / img_dir_key
        lbl_dir = dataset_root / "labels" / split

        if not img_dir.exists():
            err(f"[{split}] 이미지 폴더 없음: {img_dir}")
            issues.append(split)
            continue
        if not lbl_dir.exists():
            err(f"[{split}] 라벨 폴더 없음: {lbl_dir}")
            issues.append(split)
            continue

        images = [p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS]
        labels = list(lbl_dir.glob("*.txt"))

        ok(f"[{split}] 이미지 {len(images)}장  /  라벨 {len(labels)}개")

        # 이름 매칭 검사
        img_stems = {p.stem for p in images}
        lbl_stems = {p.stem for p in labels}
        no_label = img_stems - lbl_stems
        no_image = lbl_stems - img_stems

        if no_label:
            warn(f"[{split}] 라벨 없는 이미지 {len(no_label)}개: {sorted(no_label)[:5]} ...")
        if no_image:
            warn(f"[{split}] 이미지 없는 라벨 {len(no_image)}개: {sorted(no_image)[:5]} ...")
        if not no_label and not no_image:
            ok(f"[{split}] 이미지-라벨 완전 매칭")

        # 빈 라벨 파일 개수
        empty = [p for p in labels if p.stat().st_size == 0]
        if empty:
            info(f"[{split}] 빈 라벨(배경 이미지) {len(empty)}개")

        # 클래스 ID 범위 샘플 검사 (처음 200개)
        if isinstance(nc, int):
            bad_cls = []
            for lp in labels[:200]:
                for line in lp.read_text().splitlines():
                    parts = line.strip().split()
                    if parts and int(parts[0]) >= nc:
                        bad_cls.append(f"{lp.name}: class {parts[0]}")
            if bad_cls:
                err(f"[{split}] 클래스 ID 범위 초과 (nc={nc}): {bad_cls[:3]}")
                issues.append(split)
            else:
                ok(f"[{split}] 클래스 ID 범위 정상 (샘플 200개 검사)")

    if issues:
        err(f"데이터셋 오류 발견: {issues} — 학습을 중단합니다.")
        sys.exit(1)

    print()


def print_train_plan(args):
    sep("학습 설정")
    info(f"모델       : {args.model}")
    info(f"데이터     : {args.data}")
    info(f"에폭       : {args.epochs}")
    info(f"이미지 크기: {args.imgsz}px")
    info(f"배치 사이즈: {args.batch}")
    info(f"디바이스   : {args.device}")
    info(f"저장 폴더  : runs/detect/{args.name}")
    info(f"conf       : {args.conf}  (val confidence threshold)")
    info(f"max_det    : {args.max_det}  (이미지당 최대 bbox 수)")
    info(f"시작 시각  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()


# ── 학습 ──────────────────────────────────────────────────────
def main():
    args = parse_args()

    sep("YOLO Auto Labeling — 학습 시작")

    check_environment(args)
    check_model(args)
    check_dataset(args)
    print_train_plan(args)

    sep("학습 진행")
    start = time.time()

    model = YOLO(args.model)
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=str(PROJECT_ROOT / "runs" / "detect"),
        name=args.name,
        conf=args.conf,
        max_det=args.max_det,
    )

    elapsed = time.time() - start
    h, m, s = int(elapsed // 3600), int((elapsed % 3600) // 60), int(elapsed % 60)

    sep("학습 완료")
    ok(f"총 소요 시간 : {h}시간 {m}분 {s}초")
    ok(f"best.pt 위치 : runs/detect/{args.name}/weights/best.pt")
    ok(f"last.pt 위치 : runs/detect/{args.name}/weights/last.pt")
    info(f"다음 단계: python scripts/predict.py --model runs/detect/{args.name}/weights/best.pt")
    sep()


if __name__ == "__main__":
    main()
