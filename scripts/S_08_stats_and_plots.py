from __future__ import annotations

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


ANALYSIS_INPUT_DIR = Path("results/analysis_tables")
OUTPUT_DIR = Path("results/final_analysis")

FINAL_CSV = ANALYSIS_INPUT_DIR / "epoch_final_metrics.csv"
BEST_CSV = ANALYSIS_INPUT_DIR / "best_epoch_metrics.csv"
SUMMARY_CSV = ANALYSIS_INPUT_DIR / "summary_metrics.csv"

PIPELINE_ORDER = ["A_N0", "A_N1", "C_N0", "C_N1", "D_N0", "D_N1"]


def load_dataframes() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not FINAL_CSV.exists():
        raise FileNotFoundError(f"No existe: {FINAL_CSV}")
    if not BEST_CSV.exists():
        raise FileNotFoundError(f"No existe: {BEST_CSV}")
    if not SUMMARY_CSV.exists():
        raise FileNotFoundError(f"No existe: {SUMMARY_CSV}")

    final_df = pd.read_csv(FINAL_CSV)
    best_df = pd.read_csv(BEST_CSV)
    summary_df = pd.read_csv(SUMMARY_CSV)

    return final_df, best_df, summary_df


def order_pipeline_column(df: pd.DataFrame) -> pd.DataFrame:
    if "pipeline" in df.columns:
        df["pipeline"] = pd.Categorical(df["pipeline"], categories=PIPELINE_ORDER, ordered=True)
        df = df.sort_values(by="pipeline").reset_index(drop=True)
    return df


def save_clean_tables(final_df: pd.DataFrame, best_df: pd.DataFrame) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    final_clean = final_df[
        [
            "experiment_name",
            "pipeline",
            "visual_pipeline",
            "normalization",
            "seed",
            "epochs",
            "epoch",
            "precision",
            "recall",
            "mAP50",
            "mAP50_95",
        ]
    ].copy()

    best_clean = best_df[
        [
            "experiment_name",
            "pipeline",
            "visual_pipeline",
            "normalization",
            "seed",
            "epochs",
            "epoch",
            "precision",
            "recall",
            "mAP50",
            "mAP50_95",
        ]
    ].copy()

    final_clean = order_pipeline_column(final_clean)
    best_clean = order_pipeline_column(best_clean)

    final_clean.to_csv(OUTPUT_DIR / "final_epoch_clean_table.csv", index=False)
    best_clean.to_csv(OUTPUT_DIR / "best_epoch_clean_table.csv", index=False)

    print(f"[OK] Guardado: {OUTPUT_DIR / 'final_epoch_clean_table.csv'}")
    print(f"[OK] Guardado: {OUTPUT_DIR / 'best_epoch_clean_table.csv'}")


def flatten_pivot_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convierte columnas MultiIndex en columnas planas tipo:
    mAP50_A_N0, precision_A_N1, etc.
    """
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [
            f"{metric}_{pipeline}" if pipeline else str(metric)
            for metric, pipeline in df.columns
        ]
    return df


def build_pivot_tables(final_df: pd.DataFrame, best_df: pd.DataFrame) -> None:
    final_pivot = final_df.pivot_table(
        index="seed",
        columns="pipeline",
        values=["precision", "recall", "mAP50", "mAP50_95"],
        aggfunc="first"
    )

    best_pivot = best_df.pivot_table(
        index="seed",
        columns="pipeline",
        values=["precision", "recall", "mAP50", "mAP50_95"],
        aggfunc="first"
    )

    final_pivot = flatten_pivot_columns(final_pivot).reset_index()
    best_pivot = flatten_pivot_columns(best_pivot).reset_index()

    final_pivot.to_csv(OUTPUT_DIR / "final_epoch_pivot_table.csv", index=False)
    best_pivot.to_csv(OUTPUT_DIR / "best_epoch_pivot_table.csv", index=False)

    print(f"[OK] Guardado: {OUTPUT_DIR / 'final_epoch_pivot_table.csv'}")
    print(f"[OK] Guardado: {OUTPUT_DIR / 'best_epoch_pivot_table.csv'}")

def build_normalization_comparison_table(best_df: pd.DataFrame) -> None:
    """
    Compara N0 vs N1 dentro del mismo pipeline visual (A, C, D)
    por seed, y calcula deltas.
    """
    required_cols = [
        "visual_pipeline",
        "normalization",
        "seed",
        "precision",
        "recall",
        "mAP50",
        "mAP50_95",
    ]

    missing_cols = [c for c in required_cols if c not in best_df.columns]
    if missing_cols:
        print(f"[WARN] No se puede construir comparación N0 vs N1. Faltan columnas: {missing_cols}")
        return

    df = best_df[required_cols].copy()

    n0 = df[df["normalization"] == "N0"].copy()
    n1 = df[df["normalization"] == "N1"].copy()

    n0 = n0.rename(columns={
        "precision": "precision_N0",
        "recall": "recall_N0",
        "mAP50": "mAP50_N0",
        "mAP50_95": "mAP50_95_N0",
    })

    n1 = n1.rename(columns={
        "precision": "precision_N1",
        "recall": "recall_N1",
        "mAP50": "mAP50_N1",
        "mAP50_95": "mAP50_95_N1",
    })

    merged = pd.merge(
        n0,
        n1,
        on=["visual_pipeline", "seed"],
        how="inner"
    )

    if merged.empty:
        print("[INFO] No hay suficientes pares N0/N1 para construir comparación.")
        return

    merged["delta_precision"] = merged["precision_N1"] - merged["precision_N0"]
    merged["delta_recall"] = merged["recall_N1"] - merged["recall_N0"]
    merged["delta_mAP50"] = merged["mAP50_N1"] - merged["mAP50_N0"]
    merged["delta_mAP50_95"] = merged["mAP50_95_N1"] - merged["mAP50_95_N0"]

    ordered_cols = [
        "seed",
        "visual_pipeline",
        "precision_N0", "precision_N1", "delta_precision",
        "recall_N0", "recall_N1", "delta_recall",
        "mAP50_N0", "mAP50_N1", "delta_mAP50",
        "mAP50_95_N0", "mAP50_95_N1", "delta_mAP50_95",
    ]

    merged = merged[ordered_cols].sort_values(by=["visual_pipeline", "seed"]).reset_index(drop=True)
    merged.to_csv(OUTPUT_DIR / "normalization_comparison_by_seed.csv", index=False)

    print(f"[OK] Guardado: {OUTPUT_DIR / 'normalization_comparison_by_seed.csv'}")
    print("\n=== Comparación N0 vs N1 por seed ===")
    print(merged.to_string(index=False))
    
def get_summary_by_type(summary_df: pd.DataFrame, summary_type: str) -> pd.DataFrame:
    df = summary_df[summary_df["summary_type"] == summary_type].copy()
    df = order_pipeline_column(df)
    return df


def plot_metric_bar(summary_df: pd.DataFrame, metric: str, title: str, filename: str) -> None:
    """
    metric debe ser base sin sufijo, por ejemplo:
    precision, recall, mAP50, mAP50_95
    """
    df = get_summary_by_type(summary_df, "best_epoch")

    mean_col = f"{metric}_mean"
    std_col = f"{metric}_std"

    if mean_col not in df.columns or std_col not in df.columns:
        print(f"[WARN] Columnas no encontradas para {metric}")
        return

    plt.figure(figsize=(10, 6))
    plt.bar(df["pipeline"].astype(str), df[mean_col], yerr=df[std_col], capsize=5)
    plt.xlabel("Pipeline")
    plt.ylabel(metric)
    plt.title(title)
    plt.tight_layout()

    out_path = OUTPUT_DIR / filename
    plt.savefig(out_path, dpi=150)
    plt.close()

    print(f"[OK] Guardado: {out_path}")


def build_comparison_table(summary_df: pd.DataFrame) -> None:
    df = get_summary_by_type(summary_df, "best_epoch").copy()

    columns = [
        "pipeline",
        "precision_mean", "precision_std",
        "recall_mean", "recall_std",
        "mAP50_mean", "mAP50_std",
        "mAP50_95_mean", "mAP50_95_std",
    ]

    existing_cols = [c for c in columns if c in df.columns]
    comparison_df = df[existing_cols].copy()
    comparison_df.to_csv(OUTPUT_DIR / "best_epoch_summary_table.csv", index=False)

    print(f"[OK] Guardado: {OUTPUT_DIR / 'best_epoch_summary_table.csv'}")


def print_console_summary(summary_df: pd.DataFrame) -> None:
    df = get_summary_by_type(summary_df, "best_epoch").copy()

    cols = [
        "pipeline",
        "precision_mean", "precision_std",
        "recall_mean", "recall_std",
        "mAP50_mean", "mAP50_std",
        "mAP50_95_mean", "mAP50_95_std",
    ]
    cols = [c for c in cols if c in df.columns]

    print("\n=== Resumen best_epoch ===")
    print(df[cols].to_string(index=False))


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    final_df, best_df, summary_df = load_dataframes()

    final_df = order_pipeline_column(final_df)
    best_df = order_pipeline_column(best_df)
    summary_df = order_pipeline_column(summary_df)

    save_clean_tables(final_df, best_df)
    build_pivot_tables(final_df, best_df)
    build_normalization_comparison_table(best_df)
    build_comparison_table(summary_df)

    # Gráficas principales
    plot_metric_bar(
        summary_df,
        metric="mAP50_95",
        title="Comparación de mAP50-95 promedio por pipeline (best_epoch)",
        filename="plot_map50_95_best_epoch.png",
    )

    plot_metric_bar(
        summary_df,
        metric="mAP50",
        title="Comparación de mAP50 promedio por pipeline (best_epoch)",
        filename="plot_map50_best_epoch.png",
    )

    plot_metric_bar(
        summary_df,
        metric="precision",
        title="Comparación de Precision promedio por pipeline (best_epoch)",
        filename="plot_precision_best_epoch.png",
    )

    plot_metric_bar(
        summary_df,
        metric="recall",
        title="Comparación de Recall promedio por pipeline (best_epoch)",
        filename="plot_recall_best_epoch.png",
    )

    print_console_summary(summary_df)


if __name__ == "__main__":
    main()