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
    --device  mps          cpu / mps (Apple Silicon) / 0 (CUDA)
    --name    bird_v1      저장 폴더명
"""

import argparse
from pathlib import Path
from ultralytics import YOLO

SCRIPT_DIR = Path(__file__).resolve().parent
AUTO_DIR   = SCRIPT_DIR.parent
YAML_PATH  = AUTO_DIR / "dataset_bird.yaml"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",  default="yolov8n.pt",
                        help="베이스 모델. yolov8n/s/m/l/x.pt 중 선택")
    parser.add_argument("--data",   default=None,
                        help="dataset yaml 경로 (기본: 스크립트 상위폴더/dataset_bird.yaml)")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz",  type=int, default=320)
    parser.add_argument("--batch",  type=int, default=32)
    parser.add_argument("--device", default="mps",
                        help="mps (Apple Silicon) / cpu / 0 (CUDA)")
    parser.add_argument("--name",   default="bird_v1")
    args = parser.parse_args()

    yaml_path = Path(args.data) if args.data else YAML_PATH

    if not yaml_path.exists():
        print(f"[오류] yaml 없음: {yaml_path}")
        print(f"  python scripts/prepare_bird_dataset.py")
        return

    # 베이스 모델 경로: auto/ 에 있으면 사용, 없으면 Ultralytics가 자동 다운로드
    base_model = AUTO_DIR / args.model
    if not base_model.exists():
        base_model = args.model   # 자동 다운로드

    print(f"\n{'='*55}")
    print(f"  Bird 전용 모델 학습")
    print(f"  베이스  : {base_model}")
    print(f"  데이터  : {yaml_path}")
    print(f"  클래스  : bird / drone / uncertain (3개)")
    print(f"  Epochs  : {args.epochs}")
    print(f"  저장    : runs/detect/{args.name}/weights/best.pt")
    print(f"{'='*55}\n")

    model = YOLO(str(base_model))

    model.train(
        data          = str(yaml_path),
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
        workers       = 2,             # MPS에선 적게 잡는게 안정적
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
