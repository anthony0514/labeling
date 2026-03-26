"""
자동 라벨링 (Auto Labeling) 스크립트
학습된 모델로 unlabeled 이미지에 bbox txt를 자동 생성합니다.
"""
import argparse
from pathlib import Path
from ultralytics import YOLO

# 프로젝트 루트 = scripts/ 의 상위 폴더 (어디서 실행해도 항상 고정)
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def parse_args():
    parser = argparse.ArgumentParser(description="YOLOv8 자동 라벨링")
    parser.add_argument("--model", type=str,
                        default=str(PROJECT_ROOT / "runs/detect/train/weights/best.pt"),
                        help="학습된 모델 가중치 경로")
    parser.add_argument("--source", type=str, default=str(PROJECT_ROOT / "unlabeled"),
                        help="라벨링할 이미지 폴더 경로")
    parser.add_argument("--conf", type=float, default=0.5,
                        help="Confidence threshold (권장: 0.4~0.6)")
    parser.add_argument("--iou", type=float, default=0.45,
                        help="NMS IoU threshold")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="입력 이미지 크기")
    parser.add_argument("--device", type=str, default="mps",
                        help="디바이스 (mps / cpu / 0)")
    parser.add_argument("--name", type=str, default="predict_v3",
                        help="결과 저장 폴더 이름 (runs/detect/<name>). 이미 존재하면 _2, _3, ... 자동 부여")
    parser.add_argument("--save_img", action="store_true",
                        help="bbox 시각화 이미지 저장 여부")
    return parser.parse_args()


def resolve_unique_name(detect_dir: Path, name: str) -> str:
    """runs/detect/ 내에서 겹치지 않는 폴더 이름을 반환합니다."""
    if not (detect_dir / name).exists():
        return name
    counter = 2
    while (detect_dir / f"{name}_{counter}").exists():
        counter += 1
    return f"{name}_{counter}"


def main():
    args = parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"[ERROR] 모델 파일을 찾을 수 없습니다: {model_path}")
        print("       먼저 scripts/train.py 로 모델을 학습하세요.")
        return

    source_path = Path(args.source)
    if not source_path.exists():
        print(f"[ERROR] 소스 경로를 찾을 수 없습니다: {source_path}")
        return

    detect_dir = PROJECT_ROOT / "runs" / "detect"
    run_name = resolve_unique_name(detect_dir, args.name)
    if run_name != args.name:
        print(f"[INFO] '{args.name}' 폴더가 이미 존재합니다. '{run_name}' 으로 저장합니다.")

    print(f"모델: {args.model}")
    print(f"소스: {args.source}")
    print(f"Confidence threshold: {args.conf}")
    print(f"출력 폴더: runs/detect/{run_name}")

    model = YOLO(str(model_path))
    results = model.predict(
        source=str(source_path),
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        device=args.device,
        save_txt=True,
        save_conf=False,
        save=args.save_img,
        project=str(detect_dir),
        name=run_name,
        exist_ok=True,
    )

    label_dir = detect_dir / run_name / "labels"
    print(f"\n자동 라벨링 완료.")
    print(f"생성된 txt 파일 위치: {label_dir}")


if __name__ == "__main__":
    main()
