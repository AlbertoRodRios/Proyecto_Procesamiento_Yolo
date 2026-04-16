from __future__ import annotations

from ultralytics import YOLO
from scripts.S_03_custom_trainers import StandardizedDetectionValidator


def main() -> None:
    model = YOLO("results/detect_experiments/pipeline_B_e50_gpu_seed42/weights/best.pt")

    metrics = model.val(
        data="data/pipeline_A/dataset.yaml",
        imgsz=640,
        batch=16,
        device=0,
        workers=4,
        validator=StandardizedDetectionValidator,
        split="val",   # o "test" si luego defines evaluación final ahí
        project="results/detect_experiments",
        name="pipeline_B_eval_val_seed42",
        exist_ok=False,
    )

    print("\n=== Métricas de evaluación B ===")
    print(metrics)


if __name__ == "__main__":
    main()