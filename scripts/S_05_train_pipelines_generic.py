from __future__ import annotations

from pathlib import Path
from ultralytics import YOLO
from scripts.S_03_custom_trainers import StandardizedDetectionTrainer

# =========================
# CONFIGURACIÓN EXPERIMENTO
# =========================
VISUAL_PIPELINE = "B"   # A, C, D
NORMALIZATION = "N1"    # N0, N1
EPOCHS = 50
SEED = 42
DEVICE = 0
BATCH = 16
IMGSZ = 640
WORKERS = 4
PRETRAINED_MODEL = "yolov8n.pt"
PROJECT_DIR = "results/detect_experiments"

VALID_VISUAL_PIPELINES = {"A", "B", "C"}
VALID_NORMALIZATIONS = {"N0", "N1"}


def build_dataset_path(visual_pipeline: str) -> Path:
    return Path(f"data/pipeline_{visual_pipeline}/dataset.yaml")


def build_run_name(
    visual_pipeline: str,
    normalization: str,
    epochs: int,
    seed: int
) -> str:
    return f"pipeline_{visual_pipeline}_{normalization}_e{epochs}_gpu_seed{seed}"


def validate_config() -> None:
    if VISUAL_PIPELINE not in VALID_VISUAL_PIPELINES:
        raise ValueError(
            f"VISUAL_PIPELINE inválido: {VISUAL_PIPELINE}. "
            f"Usa uno de {sorted(VALID_VISUAL_PIPELINES)}"
        )

    if NORMALIZATION not in VALID_NORMALIZATIONS:
        raise ValueError(
            f"NORMALIZATION inválido: {NORMALIZATION}. "
            f"Usa uno de {sorted(VALID_NORMALIZATIONS)}"
        )

    dataset_yaml = build_dataset_path(VISUAL_PIPELINE)
    if not dataset_yaml.exists():
        raise FileNotFoundError(f"No existe el dataset: {dataset_yaml}")

    stats_path = Path("data/channel_stats.json")
    if NORMALIZATION == "N1" and not stats_path.exists():
        raise FileNotFoundError(f"No existe el archivo de stats: {stats_path}")


def main() -> None:
    validate_config()

    dataset_yaml = build_dataset_path(VISUAL_PIPELINE)
    run_name = build_run_name(VISUAL_PIPELINE, NORMALIZATION, EPOCHS, SEED)

    print("=== Configuración del experimento ===")
    print(f"Visual pipeline : {VISUAL_PIPELINE}")
    print(f"Normalization   : {NORMALIZATION}")
    print(f"Dataset         : {dataset_yaml}")
    print(f"Run name        : {run_name}")
    print(f"Epochs          : {EPOCHS}")
    print(f"Seed            : {SEED}")
    print(f"Device          : {DEVICE}")
    print("====================================")

    model = YOLO(PRETRAINED_MODEL)

    train_kwargs = dict(
        data=str(dataset_yaml),
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        project=PROJECT_DIR,
        name=run_name,
        workers=WORKERS,
        device=DEVICE,
        pretrained=True,
        seed=SEED,
        exist_ok=False,
    )

    if NORMALIZATION == "N1":
        train_kwargs["trainer"] = StandardizedDetectionTrainer

    model.train(**train_kwargs)


if __name__ == "__main__":
    main()