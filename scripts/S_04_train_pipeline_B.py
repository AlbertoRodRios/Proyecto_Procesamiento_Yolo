from __future__ import annotations

from ultralytics import YOLO
from scripts.S_03_custom_trainers import StandardizedDetectionTrainer


def main() -> None:
    model = YOLO("yolov8n.pt")

    model.train(
        data="data/pipeline_A/dataset.yaml",
        epochs=50,
        imgsz=640,
        batch=16,
        project="results/detect_experiments",
        name="pipeline_B_e50_gpu_seed7",
        trainer=StandardizedDetectionTrainer,
        workers=4,
        device=0,
        pretrained=True,
        seed=7,
        exist_ok=False,
    )


if __name__ == "__main__":
    main()