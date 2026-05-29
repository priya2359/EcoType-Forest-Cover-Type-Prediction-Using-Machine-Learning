# filename: src/model_trainer.py
# purpose:  Train all 6 models with MLflow logging; save each immediately after training
# version:  3.0

import logging
import os
import joblib
import mlflow
import yaml
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

from src.xgb_wrapper import XGBWrapper
from src.model_evaluator import evaluate, plot_confusion_matrix

logger = logging.getLogger(__name__)


def get_model(name: str, config: dict):
    params = config["models"][name]
    if name == "random_forest":
        return RandomForestClassifier(**params)
    elif name == "extra_trees":
        return ExtraTreesClassifier(**params)
    elif name == "decision_tree":
        return DecisionTreeClassifier(**params)
    elif name == "logistic_regression":
        return LogisticRegression(**params)
    elif name == "knn":
        return KNeighborsClassifier(**{k: v for k, v in params.items()})
    elif name == "xgboost":
        return XGBWrapper(**params)
    else:
        raise ValueError(f"Unknown model: {name}")


def train_model(model, X_train, y_train):
    model.fit(X_train, y_train)
    return model


def log_run_to_mlflow(
    model_name: str,
    model,
    metrics: dict,
    params: dict,
    cm_save_path: str = None,
    class_map: dict = None,
) -> str:
    with mlflow.start_run(run_name=model_name) as run:
        mlflow.log_params(params)
        mlflow.log_metric("accuracy", metrics["accuracy"])
        mlflow.log_metric("macro_f1", metrics["macro_f1"])
        mlflow.log_metric("weighted_f1", metrics["weighted_f1"])

        # Per-class F1 scores keyed by numeric class ID
        if class_map and "classification_report" in metrics:
            cr = metrics["classification_report"]
            for cls_id, cls_name in class_map.items():
                if cls_name in cr and isinstance(cr[cls_name], dict):
                    mlflow.log_metric(f"f1_class_{cls_id}", cr[cls_name]["f1-score"])

        if cm_save_path and os.path.exists(cm_save_path):
            mlflow.log_artifact(cm_save_path)

        # API loads joblib pkl — do not log model artifact to mlflow
        return run.info.run_id


def save_model(model, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    logger.info("Saved model: %s", path)
