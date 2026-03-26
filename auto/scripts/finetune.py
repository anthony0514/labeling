"""
finetune.py
============
현재 best.pt 에서 파인튜닝하는 스크립트.

사용법:
    cd ~/labeling/auto
    source venv/bin/activate

    # 방법 1: 파인튜닝 (추천 - 기존 모델 보존하며 추가 학습)
    python scripts/finetune.py --mode finetune

    # 방법 2: 전체 재학습 (데이터 대폭 늘었을 때)
    python scripts/finetune.py --mode retrain

    # 방법 3: 헤드만 학습 (데이터 적을 때 / 빠르게)
    python scripts/finetune.py --mode head-only

    # 공통 옵션
    --epochs 50       학습 epoch 수 (기본: mode별 자동)
    --imgsz 640       입력 해상도
    --batch  16       배치 크기 (GPU 메모리에 맞게 조정)
    --device cpu      cpu / 0 / mps (Apple Silicon)
"""

import argparse
from pathlib import Path
from ultralytics import YOLO

SCRIPT_DIR  = Path(__file__).resolve().parent
AUTO_DIR    = SCRIPT_DIR.parent
MODEL_PATH  = AUTO_DIR / "runs/detect/train/weights/best.pt"
YAML_PATH   = AUTO_DIR / "dataset.yaml"

# mode별 기본 epoch
DEFAULT_EPOCHS = {
    "finetune":  50,
    "retrain":  150,
    "head-only": 30,
}

# mode별 freeze 레이어 수 (YOLOv8n 기준)
# backbone = layers 0~9, neck+head = 10~
FREEZE_LAYERS = {
    "finetune":  0,    # 전체 학습
    "retrain":   0,    # 전체 학습
    "head-only": 10,   # backbone 고정
}

# mode별 학습률
LEARNING_RATE = {
    "finetune":  0.001,   # 작게 — 기존 가중치 보존
    "retrain":   0.01,    # 기본값
    "head-only": 0.005,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",   choices=["finetune", "retrain", "head-only"],
                        default="finetune")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--imgsz",  type=int, default=640)
    parser.add_argument("--batch",  type=int, default=16)
    parser.add_argument("--device", type=str, default="mps",
                        help="cpu / 0 (cuda:0) / mps (Apple Silicon)")
    parser.add_argument("--name",   type=str, default=None,
                        help="저장 폴더명 (기본: finetune_v1 등)")
    args = parser.parse_args()

    mode    = args.mode
    epochs  = args.epochs or DEFAULT_EPOCHS[mode]
    freeze  = FREEZE_LAYERS[mode]
    lr0     = LEARNING_RATE[mode]
    name    = args.name or f"{mode}_v1"

    print(f"\n{'='*50}")
    print(f"모드   : {mode}")
    print(f"베이스  : {MODEL_PATH}")
    print(f"데이터  : {YAML_PATH}")
    print(f"Epoch  : {epochs}")
    print(f"Freeze : 앞 {freeze}개 레이어")
    print(f"LR     : {lr0}")
    print(f"저장   : runs/detect/{name}")
    print(f"{'='*50}\n")

    model = YOLO(str(MODEL_PATH))

    train_args = dict(
        data      = str(YAML_PATH),
        epochs    = epochs,
        imgsz     = args.imgsz,
        batch     = args.batch,
        device    = args.device,
        lr0       = lr0,
        lrf       = 0.1,           # 최종 lr = lr0 * lrf
        warmup_epochs = 3,
        freeze    = freeze,        # 0이면 전체 학습
        patience  = 20,            # Early stopping
        save      = True,
        project   = str(AUTO_DIR / "runs/detect"),
        name      = name,
        exist_ok  = False,
        # 데이터 증강 (파인튜닝 시 약하게)
        hsv_h     = 0.015,
        hsv_s     = 0.4,
        hsv_v     = 0.4,
        degrees   = 5.0 if mode == "finetune" else 10.0,
        fliplr    = 0.5,
        mosaic    = 0.5 if mode == "finetune" else 1.0,
        mixup     = 0.0 if mode == "head-only" else 0.1,
    )

    results = model.train(**train_args)

    # 결과 출력
    best = AUTO_DIR / f"runs/detect/{name}/weights/best.pt"
    print(f"\n✅ 학습 완료!")
    print(f"   best.pt → {best}")
    print(f"\n review_gui.py 에서 MODEL_PATH 를 변경하면 새 모델로 검수 가능:")
    print(f"   MODEL_PATH = Path('{best}')")


if __name__ == "__main__":
    main()
