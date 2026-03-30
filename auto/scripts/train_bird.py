"""
train_bird.py
=============
bird / drone / uncertain 3-class 전용 YOLOv8 모델 신규 학습.
기존 equipment 모델(best.pt)과 완전히 별개.

사용법:
    cd ~/labeling/auto
    source venv/bin/activate
    python scripts/train_bird.py [옵션]

옵션:
    --model   yolov8n.pt   베이스 모델 (n/s/m/l/x), 기본 yolov8n.pt
    --epochs  100          기본 100
    --imgsz   640
    --batch   16
    --device  auto         cpu / mps (Apple Silicon) / cuda — 기본: 자동 감지
    --name    bird_v1      저장 폴더명
"""

import argparse
from pathlib import Path

import torch
from ultralytics import YOLO


def get_device() -> str:
    """환경에 따라 최적 디바이스 자동 감지 (mps → cuda → cpu)."""
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

SCRIPT_DIR = Path(__file__).resolve().parent
AUTO_DIR   = SCRIPT_DIR.parent
YAML_PATH  = AUTO_DIR / "dataset_bird.yaml"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",  default="yolov8n.pt",
                        help="베이스 모델. yolov8n/s/m/l/x.pt 중 선택")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz",  type=int, default=320)  # 416→320: 속도 ~1.7배↑
    parser.add_argument("--batch",  type=int, default=32)   # 16→32: iteration 절반
    parser.add_argument("--device", default=get_device(),
                        help="디바이스 (자동 감지 / mps / cuda / cpu / 0)")
    parser.add_argument("--name",   default="bird_v1")
    args = parser.parse_args()

    if not YAML_PATH.exists():
        print(f"[오류] dataset_bird.yaml 없음. 먼저 실행하세요:")
        print(f"  python scripts/prepare_bird_dataset.py")
        return

    # 베이스 모델 경로: auto/ 에 있으면 사용, 없으면 Ultralytics가 자동 다운로드
    base_model = AUTO_DIR / args.model
    if not base_model.exists():
        base_model = args.model   # 자동 다운로드

    print(f"\n{'='*55}")
    print(f"  Bird 전용 모델 학습")
    print(f"  베이스  : {base_model}")
    print(f"  데이터  : {YAML_PATH}")
    print(f"  클래스  : bird / drone / uncertain (3개)")
    print(f"  Epochs  : {args.epochs}")
    print(f"  저장    : runs/detect/{args.name}/weights/best.pt")
    print(f"{'='*55}\n")

    model = YOLO(str(base_model))

    model.train(
        data          = str(YAML_PATH),
        epochs        = args.epochs,
        imgsz         = args.imgsz,
        batch         = args.batch,
        device        = args.device,
        lr0           = 0.01,
        lrf           = 0.01,
        warmup_epochs = 3,
        patience      = 20,
        save          = True,
        cache         = "disk",         # RAM 대신 디스크 캐시 → 메모리 절약
        workers       = 2,             # 적게 잡는게 안정적
        project       = str(AUTO_DIR / "runs/detect"),
        name          = args.name,
        exist_ok      = True,
        # 증강 경량화 (속도 우선)
        mosaic        = 0.5,           # 1.0 → 0.5
        hsv_h         = 0.015,
        hsv_s         = 0.5,
        hsv_v         = 0.3,
        degrees       = 5.0,
        fliplr        = 0.5,
        flipud        = 0.0,           # 제거
        scale         = 0.3,
        mixup         = 0.0,           # 제거
    )

    best = AUTO_DIR / f"runs/detect/{args.name}/weights/best.pt"
    print(f"\n✅ 학습 완료!")
    print(f"   best.pt → {best}")
    print(f"\n review_gui.py 에서 config_bird 선택 시 이 모델을 쓰려면:")
    print(f"   MODEL_PATH_BIRD = Path('{best}')")


if __name__ == "__main__":
    main()
