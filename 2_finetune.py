"""
YOLOv8n 코 탐지기 파인튜닝
독립 실행 버전 (src 폴더 불필요)

실행:
    python 2_finetune.py --data_yaml D:/pawId/dog-nose-detector.yolov8/data.yaml
"""

import argparse
from pathlib import Path


def finetune(data_yaml, epochs, device, batch, project, name):
    try:
        from ultralytics import YOLO
    except ImportError:
        print("pip install ultralytics 를 먼저 실행하세요.")
        return

    print("=" * 50)
    print("YOLOv8n 코 탐지기 파인튜닝 시작")
    print("  data_yaml : " + data_yaml)
    print("  epochs    : " + str(epochs))
    print("  device    : " + device)
    print("=" * 50)

    model = YOLO("yolov8n.pt")  # COCO 사전학습 가중치 자동 다운로드

    results = model.train(
        data      = data_yaml,
        epochs    = epochs,
        imgsz     = 640,
        batch     = batch,
        device    = device,
        project   = project,
        name      = name,
        patience  = 15,
        save_period = 10,
        pretrained  = True,
        exist_ok    = True,
    )

    best_path = str(Path(project) / name / "weights" / "best.pt")
    print("\n파인튜닝 완료!")
    print("최적 가중치: " + best_path)

    # 성능 평가
    metrics = model.val(data=data_yaml, device=device)
    map50 = float(metrics.box.map50)
    print("\n[평가 결과]")
    print("  mAP50     : " + str(round(map50, 4)) + "  (목표: 0.85 이상)")
    print("  mAP50-95  : " + str(round(float(metrics.box.map), 4)))
    print("  Precision : " + str(round(float(metrics.box.mp), 4)))
    print("  Recall    : " + str(round(float(metrics.box.mr), 4)))

    if map50 >= 0.85:
        print("\n목표 달성! mAP50 >= 0.85")
        print("다음 단계: app.py 에 YOLOv8 통합")
    else:
        print("\n목표 미달. 아래 방법 시도:")
        print("  1. --epochs 100 으로 늘리기")
        print("  2. 라벨링 데이터 추가")

    return best_path


def main():
    parser = argparse.ArgumentParser(description="YOLOv8n 파인튜닝")
    parser.add_argument("--data_yaml", required=True, help="Roboflow data.yaml 경로")
    parser.add_argument("--epochs",    type=int,   default=50)
    parser.add_argument("--batch",     type=int,   default=16)
    parser.add_argument("--device",    default="cuda", help="cuda | cpu")
    parser.add_argument("--project",   default="D:/pawId/runs/detect")
    parser.add_argument("--name",      default="nose_detector_v1")
    args = parser.parse_args()

    finetune(
        data_yaml = args.data_yaml,
        epochs    = args.epochs,
        device    = args.device,
        batch     = args.batch,
        project   = args.project,
        name      = args.name,
    )


if __name__ == "__main__":
    main()
