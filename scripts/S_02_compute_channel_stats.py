from __future__ import annotations

from pathlib import Path
import json
import math

import cv2
import numpy as np


def load_image_as_rgb_float(image_path: Path) -> np.ndarray:
    """
    Carga una imagen con OpenCV, la convierte de BGR a RGB
    y la escala a float32 en [0, 1].
    """
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError(f"No se pudo leer la imagen: {image_path}")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32) / 255.0
    return image


def collect_image_paths(images_dir: Path) -> list[Path]:
    """
    Recolecta imágenes compatibles dentro del directorio.
    """
    valid_suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    image_paths = [
        path for path in images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in valid_suffixes
    ]
    image_paths.sort()

    if not image_paths:
        raise ValueError(f"No se encontraron imágenes en: {images_dir}")

    return image_paths


def compute_channel_stats(images_dir: Path) -> dict:
    """
    Calcula mean y std global por canal usando todos los píxeles
    de todas las imágenes del directorio.
    """
    image_paths = collect_image_paths(images_dir)

    channel_sum = np.zeros(3, dtype=np.float64)
    channel_sum_sq = np.zeros(3, dtype=np.float64)
    total_pixels = 0

    for idx, image_path in enumerate(image_paths, start=1):
        image = load_image_as_rgb_float(image_path)

        # image shape: (H, W, 3)
        h, w, c = image.shape
        if c != 3:
            raise ValueError(f"La imagen no tiene 3 canales: {image_path}")

        pixels = image.reshape(-1, 3)  # (N_pixels, 3)

        channel_sum += pixels.sum(axis=0, dtype=np.float64)
        channel_sum_sq += np.square(pixels, dtype=np.float64).sum(axis=0, dtype=np.float64)
        total_pixels += pixels.shape[0]

        if idx % 200 == 0 or idx == len(image_paths):
            print(f"Procesadas {idx}/{len(image_paths)} imágenes...")

    mean = channel_sum / total_pixels
    variance = (channel_sum_sq / total_pixels) - np.square(mean)
    variance = np.maximum(variance, 0.0)  # evita negativos por error numérico
    std = np.sqrt(variance)

    return {
        "images_dir": str(images_dir.resolve()),
        "num_images": len(image_paths),
        "total_pixels": int(total_pixels),
        "mean": mean.tolist(),
        "std": std.tolist(),
    }


def save_stats(stats: dict, output_path: Path) -> None:
    """
    Guarda las estadísticas en JSON.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=4)

    print(f"\nEstadísticas guardadas en: {output_path}")


def print_stats(stats: dict) -> None:
    mean = stats["mean"]
    std = stats["std"]

    print("\n=== Estadísticas por canal (RGB) ===")
    print(f"Imágenes procesadas : {stats['num_images']}")
    print(f"Píxeles totales     : {stats['total_pixels']}")
    print(f"Mean                : [{mean[0]:.6f}, {mean[1]:.6f}, {mean[2]:.6f}]")
    print(f"Std                 : [{std[0]:.6f}, {std[1]:.6f}, {std[2]:.6f}]")


def main() -> None:
    train_images_dir = Path("data/pipeline_A/images/train")
    output_path = Path("data/channel_stats.json")

    if not train_images_dir.exists():
        raise FileNotFoundError(
            f"No existe el directorio de imágenes de entrenamiento: {train_images_dir}"
        )

    print(f"Calculando estadísticas de canal desde: {train_images_dir}")
    stats = compute_channel_stats(train_images_dir)

    print_stats(stats)
    save_stats(stats, output_path)


if __name__ == "__main__":
    main()