from __future__ import annotations

from pathlib import Path
import shutil
import yaml
import cv2


SOURCE_DATASET = Path("data/pipeline_A")
PIPELINE_B_DIR = Path("data/pipeline_B")
PIPELINE_C_DIR = Path("data/pipeline_C")

SPLITS = ["train", "val", "test"]
VALID_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def apply_bilateral_filter(image_bgr):
    """
    Pipeline B:
    Filtro bilateral para reducción de ruido preservando bordes.
    """
    return cv2.bilateralFilter(image_bgr, d=9, sigmaColor=75, sigmaSpace=75)


def apply_clahe_lab(image_bgr):
    """
    Pipeline C:
    CLAHE aplicado únicamente al canal L del espacio LAB.
    """
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)

    merged = cv2.merge((l_enhanced, a, b))
    result_bgr = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
    return result_bgr


def ensure_clean_dir_structure(output_root: Path) -> None:
    """
    Crea la estructura base del dataset de salida.
    """
    for split in SPLITS:
        (output_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_root / "labels" / split).mkdir(parents=True, exist_ok=True)


def collect_images(images_dir: Path) -> list[Path]:
    """
    Obtiene la lista de imágenes válidas dentro de un split.
    """
    image_paths = [
        p for p in images_dir.iterdir()
        if p.is_file() and p.suffix.lower() in VALID_SUFFIXES
    ]
    image_paths.sort()

    if not image_paths:
        raise ValueError(f"No se encontraron imágenes en: {images_dir}")

    return image_paths


def transform_split(
    source_root: Path,
    output_root: Path,
    split: str,
    transform_fn,
    transform_name: str
) -> None:
    """
    Lee imágenes del split, aplica transformación y guarda salida.
    """
    src_images_dir = source_root / "images" / split
    dst_images_dir = output_root / "images" / split

    image_paths = collect_images(src_images_dir)

    print(f"\n[{transform_name}] Procesando split: {split}")
    print(f"Total imágenes: {len(image_paths)}")

    for idx, image_path in enumerate(image_paths, start=1):
        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise ValueError(f"No se pudo leer la imagen: {image_path}")

        transformed = transform_fn(image_bgr)

        out_path = dst_images_dir / image_path.name
        ok = cv2.imwrite(str(out_path), transformed)
        if not ok:
            raise ValueError(f"No se pudo guardar la imagen transformada: {out_path}")

        if idx % 200 == 0 or idx == len(image_paths):
            print(f"  Procesadas {idx}/{len(image_paths)} imágenes...")


def copy_labels(source_root: Path, output_root: Path) -> None:
    """
    Copia labels sin modificación.
    """
    for split in SPLITS:
        src_labels_dir = source_root / "labels" / split
        dst_labels_dir = output_root / "labels" / split

        label_files = list(src_labels_dir.glob("*.txt"))
        label_files.sort()

        print(f"\nCopiando labels de split '{split}' ({len(label_files)} archivos)...")

        for label_file in label_files:
            shutil.copy2(label_file, dst_labels_dir / label_file.name)


def create_dataset_yaml(output_root: Path) -> None:
    """
    Crea dataset.yaml para el pipeline generado.
    """
    data = {
        "path": str(output_root).replace("\\", "/"),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": 20,
        "names": [
            "aeroplane", "bicycle", "bird", "boat", "bottle",
            "bus", "car", "cat", "chair", "cow",
            "diningtable", "dog", "horse", "motorbike", "person",
            "pottedplant", "sheep", "sofa", "train", "tvmonitor"
        ]
    }

    yaml_path = output_root / "dataset.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)

    print(f"\n[OK] dataset.yaml generado en: {yaml_path}")


def build_pipeline(
    source_root: Path,
    output_root: Path,
    transform_fn,
    transform_name: str
) -> None:
    """
    Construye un pipeline completo.
    """
    print(f"\n=== Construyendo {transform_name} ===")
    print(f"Origen : {source_root}")
    print(f"Destino: {output_root}")

    ensure_clean_dir_structure(output_root)
    copy_labels(source_root, output_root)

    for split in SPLITS:
        transform_split(
            source_root=source_root,
            output_root=output_root,
            split=split,
            transform_fn=transform_fn,
            transform_name=transform_name
        )

    create_dataset_yaml(output_root)
    print(f"\n=== {transform_name} completado ===")


def validate_source_dataset(source_root: Path) -> None:
    """
    Verifica que pipeline_A tenga la estructura esperada.
    """
    required_paths = [
        source_root / "images" / "train",
        source_root / "images" / "val",
        source_root / "images" / "test",
        source_root / "labels" / "train",
        source_root / "labels" / "val",
        source_root / "labels" / "test",
    ]

    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(f"No existe la ruta requerida: {path}")


def main() -> None:
    validate_source_dataset(SOURCE_DATASET)

    # Pipeline C: bilateral
    build_pipeline(
        source_root=SOURCE_DATASET,
        output_root=PIPELINE_B_DIR,
        transform_fn=apply_bilateral_filter,
        transform_name="Pipeline B (Bilateral)"
    )

    # Pipeline D: CLAHE
    build_pipeline(
        source_root=SOURCE_DATASET,
        output_root=PIPELINE_C_DIR,
        transform_fn=apply_clahe_lab,
        transform_name="Pipeline C (CLAHE)"
    )

    print("\n[OK] Se generaron correctamente pipeline_B y pipeline_C.")


if __name__ == "__main__":
    main()