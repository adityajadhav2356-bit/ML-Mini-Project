"""Comprehensive Model Evaluation and Visualizations.

Computes accuracy, precision, recall, F1-score, confusion matrices, and ROC curves
for all 4 models, generating comparative tables and publication-grade figures.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

DEFAULT_REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"
FIGURES_DIR = DEFAULT_REPORTS_DIR / "figures"
DEFAULT_MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def evaluate_all_models(
    models: Dict[str, object],
    X_test_vec: np.ndarray,
    y_test: pd.Series | np.ndarray,
    figures_dir: Path | str = FIGURES_DIR,
    models_dir: Path | str = DEFAULT_MODELS_DIR,
) -> Tuple[pd.DataFrame, Dict[str, dict]]:
    """Evaluate all 4 models, generate comparison tables, and save plots.

    Parameters
    ----------
    models : Dict[str, object]
        Trained model instances keyed by algorithm name.
    X_test_vec : np.ndarray
        Test TF-IDF feature matrix.
    y_test : pd.Series | np.ndarray
        Ground truth test labels (0=ham, 1=spam).
    figures_dir : Path | str
        Destination directory for generated visualization figures.
    models_dir : Path | str
        Directory to export metrics CSV and JSON.

    Returns
    -------
    Tuple[pd.DataFrame, Dict[str, dict]]
        Summary DataFrame and detailed metrics dictionary.
    """
    fig_path = Path(figures_dir)
    fig_path.mkdir(parents=True, exist_ok=True)
    models_path = Path(models_dir)
    models_path.mkdir(parents=True, exist_ok=True)

    summary_records = []
    detailed_metrics = {}
    roc_data = {}
    cm_dict = {}

    for name, model in models.items():
        y_pred = model.predict(X_test_vec)

        # Probabilities handling
        has_proba = hasattr(model, "predict_proba")
        if has_proba:
            y_prob = model.predict_proba(X_test_vec)[:, 1]
            try:
                auc_score = float(roc_auc_score(y_test, y_prob))
                fpr, tpr, _ = roc_curve(y_test, y_prob)
                roc_data[name] = (fpr, tpr, auc_score)
            except Exception:
                auc_score = None
        else:
            auc_score = None

        acc = float(accuracy_score(y_test, y_pred))
        prec = float(precision_score(y_test, y_pred, pos_label=1, zero_division=0))
        rec = float(recall_score(y_test, y_pred, pos_label=1, zero_division=0))
        f1 = float(f1_score(y_test, y_pred, pos_label=1, zero_division=0))
        cm = confusion_matrix(y_test, y_pred)
        cm_dict[name] = cm

        tn, fp, fn, tp = cm.ravel()

        record = {
            "Algorithm": name,
            "Accuracy": round(acc, 4),
            "Precision (Spam)": round(prec, 4),
            "Recall (Spam)": round(rec, 4),
            "F1-Score (Spam)": round(f1, 4),
            "ROC-AUC": round(auc_score, 4) if auc_score is not None else "N/A",
            "True Negatives": int(tn),
            "False Positives": int(fp),
            "False Negatives": int(fn),
            "True Positives": int(tp),
        }
        summary_records.append(record)

        detailed_metrics[name] = {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
            "roc_auc": auc_score,
            "confusion_matrix": cm.tolist(),
            "has_predict_proba": has_proba,
        }

    summary_df = pd.DataFrame(summary_records)

    # Save summary tables
    csv_file = models_path / "metrics_summary.csv"
    summary_df.to_csv(csv_file, index=False)

    json_file = models_path / "evaluation_metrics.json"
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(detailed_metrics, f, indent=2)

    print(f"[Evaluation] Metrics summary saved to: {csv_file}")

    # Generate Visualizations
    _plot_metric_comparison(summary_df, fig_path / "model_comparison_bar.png")
    _plot_confusion_matrices(cm_dict, fig_path / "confusion_matrices.png")
    if roc_data:
        _plot_roc_curves(roc_data, fig_path / "roc_curves.png")

    return summary_df, detailed_metrics


def _plot_metric_comparison(df: pd.DataFrame, output_path: Path) -> None:
    """Generate side-by-side grouped bar plot of standard metrics."""
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=300)

    metrics = ["Accuracy", "Precision (Spam)", "Recall (Spam)", "F1-Score (Spam)"]
    x = np.arange(len(df))
    width = 0.2

    colors = ["#2563eb", "#10b981", "#f59e0b", "#8b5cf6"]

    for idx, (metric, color) in enumerate(zip(metrics, colors)):
        values = df[metric].values
        rects = ax.bar(x + (idx - 1.5) * width, values, width, label=metric, color=color, alpha=0.9, edgecolor="none")
        for rect in rects:
            height = rect.get_height()
            ax.annotate(
                f"{height:.2%}",
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=7.5,
                rotation=0,
            )

    ax.set_ylabel("Score (0.0 - 1.0)", fontsize=11, fontweight="bold")
    ax.set_title("Machine Learning Algorithms Benchmark: Spam Message Detection", fontsize=13, fontweight="bold", pad=14)
    ax.set_xticks(x)
    ax.set_xticklabels(df["Algorithm"], fontsize=10, fontweight="bold")
    ax.set_ylim(0.70, 1.05)
    ax.legend(loc="lower right", frameon=True, fontsize=9.5)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[Evaluation] Comparison bar chart generated: {output_path}")


def _plot_confusion_matrices(cm_dict: Dict[str, np.ndarray], output_path: Path) -> None:
    """Plot 2x2 grid of confusion matrices using Seaborn heatmap."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 9), dpi=300)
    axes = axes.flatten()

    labels = ["Ham (0)", "Spam (1)"]
    color_palettes = ["Blues", "Greens", "Purples", "Oranges"]

    for idx, (name, cm) in enumerate(cm_dict.items()):
        ax = axes[idx]
        cmap = color_palettes[idx % len(color_palettes)]
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap=cmap,
            cbar=False,
            ax=ax,
            xticklabels=labels,
            yticklabels=labels,
            annot_kws={"size": 13, "weight": "bold"},
        )
        ax.set_title(f"{name}\nConfusion Matrix", fontsize=11, fontweight="bold", pad=8)
        ax.set_xlabel("Predicted Label", fontsize=9.5, fontweight="bold")
        ax.set_ylabel("Actual Ground Truth", fontsize=9.5, fontweight="bold")

    plt.suptitle("Confusion Matrices across Classifiers (Test Partition)", fontsize=14, fontweight="bold", y=0.99)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[Evaluation] Confusion matrices figure generated: {output_path}")


def _plot_roc_curves(roc_data: Dict[str, Tuple[np.ndarray, np.ndarray, float]], output_path: Path) -> None:
    """Plot overlaid Receiver Operating Characteristic (ROC) curves."""
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    colors = ["#2563eb", "#10b981", "#8b5cf6", "#f59e0b"]

    for idx, (name, (fpr, tpr, auc_val)) in enumerate(roc_data.items()):
        color = colors[idx % len(colors)]
        label_text = f"{name} (AUC = {auc_val:.4f})" if auc_val else name
        ax.plot(fpr, tpr, color=color, lw=2, label=label_text)

    ax.plot([0, 1], [0, 1], color="#9ca3af", lw=1.5, linestyle="--", label="Random Chance (AUC = 0.5000)")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate (FPR)", fontsize=10.5, fontweight="bold")
    ax.set_ylabel("True Positive Rate (TPR / Recall)", fontsize=10.5, fontweight="bold")
    ax.set_title("Receiver Operating Characteristic (ROC) Comparison", fontsize=12, fontweight="bold", pad=12)
    ax.legend(loc="lower right", frameon=True, fontsize=9.5)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[Evaluation] ROC curves figure generated: {output_path}")
