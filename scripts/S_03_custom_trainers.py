from __future__ import annotations

from pathlib import Path
import json

import torch
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.models.yolo.detect.val import DetectionValidator


def load_channel_stats(stats_path: str | Path) -> tuple[list[float], list[float]]:
    stats_path = Path(stats_path)

    if not stats_path.exists():
        raise FileNotFoundError(f"No existe el archivo de estadísticas: {stats_path}")

    with open(stats_path, "r", encoding="utf-8") as f:
        stats = json.load(f)

    mean = stats.get("mean")
    std = stats.get("std")

    if mean is None or std is None:
        raise ValueError("El archivo JSON debe contener 'mean' y 'std'.")

    if len(mean) != 3 or len(std) != 3:
        raise ValueError("mean y std deben tener exactamente 3 valores.")

    if any(s <= 0 for s in std):
        raise ValueError("Todos los valores de std deben ser mayores que cero.")

    return mean, std


def standardize_imgs(
    imgs: torch.Tensor,
    channel_mean: torch.Tensor,
    channel_std: torch.Tensor,
) -> torch.Tensor:
    """
    Aplica:
      1) reescalado defensivo a [0,1] si hiciera falta
      2) estandarización Z-score por canal
    """

    if imgs.max() > 1.5:
        imgs = imgs / 255.0

    mean = channel_mean.to(device=imgs.device, dtype=imgs.dtype)
    std = channel_std.to(device=imgs.device, dtype=imgs.dtype)

    return (imgs - mean) / std


class StandardizedDetectionValidator(DetectionValidator):
    """
    Validador que aplica la misma estandarización por canal
    en validación/test que se usa en entrenamiento.
    """

    CHANNEL_STATS_PATH = "data/channel_stats.json"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        mean, std = load_channel_stats(self.CHANNEL_STATS_PATH)
        self.channel_mean = torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1)
        self.channel_std = torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1)

        self._debug_print_done = False

    def preprocess(self, batch):
        batch = super().preprocess(batch)
        batch["img"] = standardize_imgs(batch["img"], self.channel_mean, self.channel_std)

        if not self._debug_print_done:
            imgs = batch["img"]
            batch_mean = imgs.mean(dim=(0, 2, 3))
            batch_std = imgs.std(dim=(0, 2, 3), unbiased=False)

            print("\n[DEBUG][VAL] Estandarización aplicada en validator.preprocess")
            print(
                f"[DEBUG][VAL] Mean batch aprox.: "
                f"{batch_mean[0].item():.4f}, "
                f"{batch_mean[1].item():.4f}, "
                f"{batch_mean[2].item():.4f}"
            )
            print(
                f"[DEBUG][VAL] Std batch aprox.: "
                f"{batch_std[0].item():.4f}, "
                f"{batch_std[1].item():.4f}, "
                f"{batch_std[2].item():.4f}"
            )
            self._debug_print_done = True

        return batch


class StandardizedDetectionTrainer(DetectionTrainer):
    """
    Trainer que aplica estandarización por canal en entrenamiento
    y además fuerza un validator con la misma transformación.
    """

    CHANNEL_STATS_PATH = "data/channel_stats.json"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        mean, std = load_channel_stats(self.CHANNEL_STATS_PATH)
        self.channel_mean = torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1)
        self.channel_std = torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1)

        self._debug_print_done = False

    def preprocess_batch(self, batch):
        batch = super().preprocess_batch(batch)
        batch["img"] = standardize_imgs(batch["img"], self.channel_mean, self.channel_std)

        if not self._debug_print_done:
            imgs = batch["img"]
            batch_mean = imgs.mean(dim=(0, 2, 3))
            batch_std = imgs.std(dim=(0, 2, 3), unbiased=False)

            print("\n[DEBUG][TRAIN] Estandarización aplicada en trainer.preprocess_batch")
            print(
                f"[DEBUG][TRAIN] Mean batch aprox.: "
                f"{batch_mean[0].item():.4f}, "
                f"{batch_mean[1].item():.4f}, "
                f"{batch_mean[2].item():.4f}"
            )
            print(
                f"[DEBUG][TRAIN] Std batch aprox.: "
                f"{batch_std[0].item():.4f}, "
                f"{batch_std[1].item():.4f}, "
                f"{batch_std[2].item():.4f}"
            )
            self._debug_print_done = True

        return batch

    def get_validator(self):
        """
        Fuerza el uso del validator personalizado para que val/test
        usen el mismo Z-score que train.
        """
        self.loss_names = "box_loss", "cls_loss", "dfl_loss"
        return StandardizedDetectionValidator(
            self.test_loader,
            save_dir=self.save_dir,
            args=self.args,
            _callbacks=self.callbacks,
        )