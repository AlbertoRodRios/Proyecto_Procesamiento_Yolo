from __future__ import annotations

from pathlib import Path
import shutil
import xml.etree.ElementTree as ET
import yaml


VOC_CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]


def voc_box_to_yolo(
    img_w: int,
    img_h: int,
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float
) -> tuple[float, float, float, float]:
    """
    Convierte una caja VOC (xmin, ymin, xmax, ymax)
    a formato YOLO (x_center, y_center, width, height),
    todo normalizado a [0, 1].
    """
    x_center = ((xmin + xmax) / 2.0) / img_w
    y_center = ((ymin + ymax) / 2.0) / img_h
    width = (xmax - xmin) / img_w
    height = (ymax - ymin) / img_h
    return x_center, y_center, width, height


def clamp(value: float, min_value: float = 0.0, max_value: float = 1.0) -> float:
    return max(min_value, min(value, max_value))


def validate_yolo_box(
    x_center: float,
    y_center: float,
    width: float,
    height: float
) -> bool:
    """
    Valida que la caja YOLO tenga valores coherentes.
    """
    values = [x_center, y_center, width, height]
    if any(v < 0.0 or v > 1.0 for v in values):
        return False
    if width <= 0.0 or height <= 0.0:
        return False
    return True


def parse_annotation(
    xml_path: Path,
    ignore_difficult: bool = True
) -> list[tuple[int, tuple[float, float, float, float]]]:
    """
    Lee un XML de PASCAL VOC y devuelve una lista de etiquetas YOLO:
    [(class_id, (x_center, y_center, width, height)), ...]
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    size = root.find("size")
    if size is None:
        raise ValueError(f"No se encontró la sección <size> en {xml_path}")

    width_tag = size.find("width")
    height_tag = size.find("height")
    if width_tag is None or height_tag is None:
        raise ValueError(f"Faltan width/height en {xml_path}")

    img_w = int(width_tag.text)
    img_h = int(height_tag.text)

    if img_w <= 0 or img_h <= 0:
        raise ValueError(f"Tamaño de imagen inválido en {xml_path}: {img_w}x{img_h}")

    labels: list[tuple[int, tuple[float, float, float, float]]] = []

    for obj in root.findall("object"):
        name_tag = obj.find("name")
        if name_tag is None or not name_tag.text:
            continue

        class_name = name_tag.text.strip()
        if class_name not in VOC_CLASSES:
            continue

        difficult_tag = obj.find("difficult")
        difficult = int(difficult_tag.text) if difficult_tag is not None and difficult_tag.text else 0

        if ignore_difficult and difficult == 1:
            continue

        bndbox = obj.find("bndbox")
        if bndbox is None:
            continue

        xmin_tag = bndbox.find("xmin")
        ymin_tag = bndbox.find("ymin")
        xmax_tag = bndbox.find("xmax")
        ymax_tag = bndbox.find("ymax")

        if None in (xmin_tag, ymin_tag, xmax_tag, ymax_tag):
            continue

        xmin = float(xmin_tag.text)
        ymin = float(ymin_tag.text)
        xmax = float(xmax_tag.text)
        ymax = float(ymax_tag.text)

        # Corrección defensiva por anotaciones anómalas
        xmin = max(0.0, min(xmin, img_w))
        xmax = max(0.0, min(xmax, img_w))
        ymin = max(0.0, min(ymin, img_h))
        ymax = max(0.0, min(ymax, img_h))

        if xmax <= xmin or ymax <= ymin:
            continue

        x_center, y_center, box_w, box_h = voc_box_to_yolo(
            img_w, img_h, xmin, ymin, xmax, ymax
        )

        # clamp por seguridad numérica
        x_center = clamp(x_center)
        y_center = clamp(y_center)
        box_w = clamp(box_w)
        box_h = clamp(box_h)

        if not validate_yolo_box(x_center, y_center, box_w, box_h):
            continue

        class_id = VOC_CLASSES.index(class_name)
        labels.append((class_id, (x_center, y_center, box_w, box_h)))

    return labels


def save_yolo_label(
    label_path: Path,
    labels: list[tuple[int, tuple[float, float, float, float]]]
) -> None:
    """
    Guarda un archivo .txt en formato YOLO.
    Si no hay objetos válidos, crea archivo vacío.
    """
    label_path.parent.mkdir(parents=True, exist_ok=True)

    with open(label_path, "w", encoding="utf-8") as f:
        for class_id, (x_center, y_center, width, height) in labels:
            f.write(
                f"{class_id} "
                f"{x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n"
            )


def load_split_ids(split_file: Path) -> list[str]:
    """
    Carga los ids de imagen desde train.txt, val.txt o test.txt.
    """
    if not split_file.exists():
        raise FileNotFoundError(f"No existe el archivo de partición: {split_file}")

    with open(split_file, "r", encoding="utf-8") as f:
        ids = [line.strip() for line in f if line.strip()]

    if not ids:
        raise ValueError(f"El archivo de partición está vacío: {split_file}")

    return ids


def find_image_file(jpeg_dir: Path, image_id: str) -> Path:
    """
    Busca la imagen asociada al id. VOC 2007 normalmente usa .jpg,
    pero esto permite cierta tolerancia.
    """
    candidates = [
        jpeg_dir / f"{image_id}.jpg",
        jpeg_dir / f"{image_id}.jpeg",
        jpeg_dir / f"{image_id}.png",
        jpeg_dir / f"{image_id}.JPG",
        jpeg_dir / f"{image_id}.JPEG",
        jpeg_dir / f"{image_id}.PNG",
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(f"No se encontró imagen para id '{image_id}' en {jpeg_dir}")


def process_split(
    split_name: str,
    image_ids: list[str],
    voc_root: Path,
    output_root: Path,
    ignore_difficult: bool = True
) -> dict[str, int]:
    """
    Procesa una partición completa y devuelve estadísticas.
    """
    annotations_dir = voc_root / "Annotations"
    jpeg_dir = voc_root / "JPEGImages"

    out_images_dir = output_root / "images" / split_name
    out_labels_dir = output_root / "labels" / split_name

    out_images_dir.mkdir(parents=True, exist_ok=True)
    out_labels_dir.mkdir(parents=True, exist_ok=True)

    stats = {
        "total_ids": len(image_ids),
        "processed_images": 0,
        "missing_xml": 0,
        "missing_image": 0,
        "written_labels": 0,
        "empty_labels": 0,
    }

    for image_id in image_ids:
        xml_path = annotations_dir / f"{image_id}.xml"

        if not xml_path.exists():
            stats["missing_xml"] += 1
            continue

        try:
            image_path = find_image_file(jpeg_dir, image_id)
        except FileNotFoundError:
            stats["missing_image"] += 1
            continue

        labels = parse_annotation(xml_path, ignore_difficult=ignore_difficult)

        dst_image_path = out_images_dir / image_path.name
        dst_label_path = out_labels_dir / f"{image_id}.txt"

        shutil.copy2(image_path, dst_image_path)
        save_yolo_label(dst_label_path, labels)

        stats["processed_images"] += 1
        stats["written_labels"] += 1
        if len(labels) == 0:
            stats["empty_labels"] += 1

    return stats


def create_dataset_yaml(output_root: Path, use_relative_path: bool = True) -> Path:
    """
    Crea el archivo dataset.yaml para YOLO.
    """
    path_value = "." if use_relative_path else str(output_root.resolve())

    data = {
        "path": path_value,
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(VOC_CLASSES),
        "names": VOC_CLASSES,
    }

    yaml_path = output_root / "dataset.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)

    return yaml_path


def print_stats(split_name: str, stats: dict[str, int]) -> None:
    print(f"\n[{split_name}]")
    print(f"  IDs totales        : {stats['total_ids']}")
    print(f"  Imágenes procesadas: {stats['processed_images']}")
    print(f"  XML faltantes      : {stats['missing_xml']}")
    print(f"  Imágenes faltantes : {stats['missing_image']}")
    print(f"  Labels escritos    : {stats['written_labels']}")
    print(f"  Labels vacíos      : {stats['empty_labels']}")


def main() -> None:
    # Ajusta estas rutas si tu proyecto usa otra estructura
    voc_root = Path("data/raw_voc")
    output_root = Path("data/pipeline_A")

    if not voc_root.exists():
        raise FileNotFoundError(
            f"No existe la carpeta raíz del dataset VOC2007: {voc_root}"
        )

    splits_dir = voc_root / "ImageSets" / "Main"

    train_ids = load_split_ids(splits_dir / "train.txt")
    val_ids = load_split_ids(splits_dir / "val.txt")
    test_ids = load_split_ids(splits_dir / "test.txt")

    print("Iniciando conversión de VOC2007 a formato YOLO para Pipeline A...")
    print(f"Ruta VOC origen : {voc_root}")
    print(f"Ruta salida     : {output_root}")

    train_stats = process_split(
        split_name="train",
        image_ids=train_ids,
        voc_root=voc_root,
        output_root=output_root,
        ignore_difficult=True
    )

    val_stats = process_split(
        split_name="val",
        image_ids=val_ids,
        voc_root=voc_root,
        output_root=output_root,
        ignore_difficult=True
    )

    test_stats = process_split(
        split_name="test",
        image_ids=test_ids,
        voc_root=voc_root,
        output_root=output_root,
        ignore_difficult=True
    )

    yaml_path = create_dataset_yaml(output_root, use_relative_path=True)

    print_stats("train", train_stats)
    print_stats("val", val_stats)
    print_stats("test", test_stats)

    print(f"\nArchivo YAML generado en: {yaml_path}")
    print("\nConversión completada.")
    print("Ya puedes usar este dataset con YOLO.")


if __name__ == "__main__":
    main()