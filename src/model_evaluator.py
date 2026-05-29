# filename: src/model_evaluator.py
# purpose:  Model evaluation, confusion matrix plotting, comparison DataFrame
# version:  2.0

import logging
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

logger = logging.getLogger(__name__)

# COVER_TYPE_NAMES intentionally removed — single source of truth is
# feature_config.yaml class_map. Callers must pass class_names from config.


def evaluate(model, X_test, y_test, class_names: dict = None) -> dict:
    y_pred = model.predict(X_test)
    if class_names is not None:
        target_names = [class_names[i] for i in sorted(class_names)]
    else:
        target_names = [str(i) for i in sorted(set(y_test))]
    return {
        "accuracy": round(accuracy_score(y_test, y_pred), 6),
        "macro_f1": round(f1_score(y_test, y_pred, average="macro"), 6),
        "weighted_f1": round(f1_score(y_test, y_pred, average="weighted"), 6),
        "classification_report": classification_report(
            y_test, y_pred,
            target_names=target_names,
            output_dict=True,
        ),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "y_pred": y_pred,
    }


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list[str],
    title: str,
    save_path: str,
) -> None:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
    )
    plt.title(title)
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved confusion matrix: %s", save_path)


def build_comparison_df(results_dict: dict) -> pd.DataFrame:
    rows = []
    for model_name, metrics in results_dict.items():
        rows.append({
            "model": model_name,
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "weighted_f1": metrics["weighted_f1"],
        })
    df = pd.DataFrame(rows).sort_values("macro_f1", ascending=False)
    return df


def plot_model_comparison(
    df: pd.DataFrame,
    metric: str = "macro_f1",
    save_path: str = "reports/figures/model/model_comparison_bar.png",
) -> None:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#2196F3" if i == 0 else "#90CAF9" for i in range(len(df))]
    ax.barh(df["model"], df[metric], color=colors)
    ax.set_xlabel(metric.replace("_", " ").title())
    ax.set_title(f"Model Comparison - {metric.replace('_', ' ').title()}")
    ax.invert_yaxis()
    for i, v in enumerate(df[metric]):
        ax.text(v + 0.002, i, f"{v:.4f}", va="center")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved model comparison chart: %s", save_path)
