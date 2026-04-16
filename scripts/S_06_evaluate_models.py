from __future__ import annotations

from pathlib import Path
from ultralytics import YOLO
from scripts.S_03_custom_trainers import StandardizedDetectionValidator

# =========================
# CONFIGURACIÓN EXPERIMENTO
# =========================
VISUAL_PIPELINE = "A"   # A, B, C
NORMALIZATION = "N1"    # N0, N1
EPOCHS = 50
SEED = 42
DEVICE = 0
BATCH = 16
IMGSZ = 640
WORKERS = 4
SPLIT = "test"           # val o test
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


def build_weights_path(run_name: str) -> Path:
    """
    Ajusta esta ruta si tu Ultralytics guarda en otro árbol.
    """
    return Path(f"runs/detect/results/detect_experiments/{run_name}/weights/best.pt")


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

    if SPLIT not in {"val", "test"}:
        raise ValueError("SPLIT debe ser 'val' o 'test'.")

    dataset_yaml = build_dataset_path(VISUAL_PIPELINE)
    if not dataset_yaml.exists():
        raise FileNotFoundError(f"No existe el dataset: {dataset_yaml}")

    run_name = build_run_name(VISUAL_PIPELINE, NORMALIZATION, EPOCHS, SEED)
    weights_path = build_weights_path(run_name)
    if not weights_path.exists():
        raise FileNotFoundError(f"No existe el modelo entrenado: {weights_path}")


def main() -> None:
    validate_config()

    dataset_yaml = build_dataset_path(VISUAL_PIPELINE)
    run_name = build_run_name(VISUAL_PIPELINE, NORMALIZATION, EPOCHS, SEED)
    weights_path = build_weights_path(run_name)

    print("=== Evaluación del modelo ===")
    print(f"Run name   : {run_name}")
    print(f"Weights    : {weights_path}")
    print(f"Dataset    : {dataset_yaml}")
    print(f"Split      : {SPLIT}")
    print(f"Device     : {DEVICE}")
    print("================================")

    model = YOLO(str(weights_path))

    val_kwargs = dict(
        data=str(dataset_yaml),
        imgsz=IMGSZ,
        batch=BATCH,
        device=DEVICE,
        workers=WORKERS,
        split=SPLIT,
        project=PROJECT_DIR,
        name=f"{run_name}_eval_{SPLIT}",
        exist_ok=False,
    )

    if NORMALIZATION == "N1":
        val_kwargs["validator"] = StandardizedDetectionValidator

    metrics = model.val(**val_kwargs)

    print("\n=== Métricas de evaluación ===")
    print(metrics)


if __name__ == "__main__":
    main()