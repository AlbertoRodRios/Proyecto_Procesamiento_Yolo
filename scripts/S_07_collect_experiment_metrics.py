from __future__ import annotations

from pathlib import Path
import pandas as pd


EXPERIMENTS_ROOT = Path("runs/detect/results/detect_experiments")
OUTPUT_DIR = Path("results/analysis_tables")

TARGET_COLUMNS = {
    "precision": "metrics/precision(B)",
    "recall": "metrics/recall(B)",
    "mAP50": "metrics/mAP50(B)",
    "mAP50_95": "metrics/mAP50-95(B)",
    "train_box_loss": "train/box_loss",
    "train_cls_loss": "train/cls_loss",
    "train_dfl_loss": "train/dfl_loss",
    "val_box_loss": "val/box_loss",
    "val_cls_loss": "val/cls_loss",
    "val_dfl_loss": "val/dfl_loss",
}


def parse_experiment_name(folder_name: str):
    """
    Soporta nombres como:
    pipeline_A_N0_e50_gpu_seed42
    pipeline_A_N1_e50_gpu_seed42
    pipeline_C_N0_e50_gpu_seed7
    pipeline_D_N1_e50_gpu_seed123
    """
    parts = folder_name.split("_")

    # Esperado:
    # ['pipeline', 'A', 'N0', 'e50', 'gpu', 'seed42']
    if len(parts) != 6:
        return None

    if parts[0] != "pipeline":
        return None

    visual_pipeline = parts[1]
    normalization = parts[2]
    epoch_tag = parts[3]
    gpu_tag = parts[4]
    seed_tag = parts[5]

    if visual_pipeline not in {"A", "B", "C", "D"}:
        return None

    if normalization not in {"N0", "N1"}:
        return None

    if not epoch_tag.startswith("e"):
        return None

    if gpu_tag != "gpu":
        return None

    if not seed_tag.startswith("seed"):
        return None

    seed = int(seed_tag.replace("seed", ""))
    epochs = int(epoch_tag.replace("e", ""))

    pipeline_label = f"{visual_pipeline}_{normalization}"

    return {
        "visual_pipeline": visual_pipeline,
        "normalization": normalization,
        "pipeline": pipeline_label,
        "seed": seed,
        "epochs": epochs,
    }


def load_results_csv(exp_dir: Path) -> pd.DataFrame | None:
    csv_path = exp_dir / "results.csv"
    if not csv_path.exists():
        return None

    df = pd.read_csv(csv_path)
    if df.empty:
        return None

    return df


def extract_row_metrics(
    row: pd.Series,
    exp_name: str,
    pipeline: str,
    seed: int,
    mode: str,
    visual_pipeline: str | None = None,
    normalization: str | None = None,
    epochs: int | None = None,
) -> dict:
    metrics = {
        "experiment_name": exp_name,
        "pipeline": pipeline,                 # ej. A_N0, A_N1, C_N0, C_N1
        "visual_pipeline": visual_pipeline,   # ej. A, C, D
        "normalization": normalization,       # ej. N0, N1
        "seed": seed,
        "epochs": epochs,
        "mode": mode,                         # final_epoch o best_epoch
        "epoch": int(row["epoch"]) if "epoch" in row else None,
    }

    for out_name, csv_col in TARGET_COLUMNS.items():
        metrics[out_name] = row[csv_col] if csv_col in row else None

    return metrics


def collect_metrics() -> tuple[pd.DataFrame, pd.DataFrame]:
    final_rows = []
    best_rows = []

    if not EXPERIMENTS_ROOT.exists():
        raise FileNotFoundError(f"No existe la ruta de experimentos: {EXPERIMENTS_ROOT}")

    for exp_dir in sorted(EXPERIMENTS_ROOT.iterdir()):
        if not exp_dir.is_dir():
            continue

        parsed = parse_experiment_name(exp_dir.name)
        if parsed is None:
            print(f"[INFO] Carpeta ignorada por no coincidir con patrón: {exp_dir.name}")
            continue

        pipeline = parsed["pipeline"]
        visual_pipeline = parsed["visual_pipeline"]
        normalization = parsed["normalization"]
        seed = parsed["seed"]
        epochs = parsed["epochs"]

        df = load_results_csv(exp_dir)
        if df is None:
            print(f"[INFO] No se encontró results.csv válido en: {exp_dir}")
            continue

        # Última época
        final_row = df.iloc[-1]
        final_rows.append(
            extract_row_metrics(
                final_row,
                exp_dir.name,
                pipeline,
                seed,
                mode="final_epoch",
                visual_pipeline=visual_pipeline,
                normalization=normalization,
                epochs=epochs,
            )
        )

        # Mejor época = máximo mAP50-95
        best_idx = df[TARGET_COLUMNS["mAP50_95"]].idxmax()
        best_row = df.loc[best_idx]
        best_rows.append(
            extract_row_metrics(
                best_row,
                exp_dir.name,
                pipeline,
                seed,
                mode="best_epoch",
                visual_pipeline=visual_pipeline,
                normalization=normalization,
                epochs=epochs,
            )
        )

    final_df = pd.DataFrame(final_rows)
    best_df = pd.DataFrame(best_rows)

    return final_df, best_df


def build_summary_table(df: pd.DataFrame, label: str) -> pd.DataFrame:
    metric_cols = [
        "precision",
        "recall",
        "mAP50",
        "mAP50_95",
        "train_box_loss",
        "train_cls_loss",
        "train_dfl_loss",
        "val_box_loss",
        "val_cls_loss",
        "val_dfl_loss",
    ]

    grouped = df.groupby("pipeline")[metric_cols].agg(["mean", "std"]).reset_index()

    grouped.columns = [
        "pipeline" if col[0] == "pipeline" else f"{col[0]}_{col[1]}"
        for col in grouped.columns
    ]

    grouped.insert(1, "summary_type", label)
    return grouped


def save_outputs(final_df: pd.DataFrame, best_df: pd.DataFrame) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    final_path = OUTPUT_DIR / "epoch_final_metrics.csv"
    best_path = OUTPUT_DIR / "best_epoch_metrics.csv"
    summary_path = OUTPUT_DIR / "summary_metrics.csv"

    final_df = final_df.sort_values(by=["pipeline", "seed"]).reset_index(drop=True)
    best_df = best_df.sort_values(by=["pipeline", "seed"]).reset_index(drop=True)

    summary_final = build_summary_table(final_df, label="final_epoch")
    summary_best = build_summary_table(best_df, label="best_epoch")
    summary_df = pd.concat([summary_final, summary_best], ignore_index=True)

    final_df.to_csv(final_path, index=False)
    best_df.to_csv(best_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    print(f"[OK] Guardado: {final_path}")
    print(f"[OK] Guardado: {best_path}")
    print(f"[OK] Guardado: {summary_path}")


def print_quick_view(final_df: pd.DataFrame, best_df: pd.DataFrame) -> None:
    cols = [
        "experiment_name",
        "pipeline",
        "visual_pipeline",
        "normalization",
        "seed",
        "epoch",
        "precision",
        "recall",
        "mAP50",
        "mAP50_95",
    ]

    print("\n=== Última época por experimento ===")
    print(final_df[cols].sort_values(by=["pipeline", "seed"]).to_string(index=False))

    print("\n=== Mejor época por experimento ===")
    print(best_df[cols].sort_values(by=["pipeline", "seed"]).to_string(index=False))

    final_summary = final_df.groupby("pipeline")[["precision", "recall", "mAP50", "mAP50_95"]].agg(["mean", "std"])
    best_summary = best_df.groupby("pipeline")[["precision", "recall", "mAP50", "mAP50_95"]].agg(["mean", "std"])

    print("\n=== Resumen final_epoch (mean/std) ===")
    print(final_summary)

    print("\n=== Resumen best_epoch (mean/std) ===")
    print(best_summary)


def main() -> None:
    final_df, best_df = collect_metrics()
    save_outputs(final_df, best_df)
    print_quick_view(final_df, best_df)


if __name__ == "__main__":
    main()