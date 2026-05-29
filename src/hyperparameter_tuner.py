# filename: src/hyperparameter_tuner.py
# purpose:  Optuna Bayesian optimisation (TPE) for RF, ET, XGB — smarter than random search
# version:  4.0

import logging
import os
import optuna
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

from src.xgb_wrapper import XGBWrapper

logger = logging.getLogger(__name__)

# Silence Optuna's per-trial INFO logs — only show warnings and above
optuna.logging.set_verbosity(optuna.logging.WARNING)


def _rf_objective(trial, X_train, y_train, cv, random_state):
    params = {
        "n_estimators":     trial.suggest_int("n_estimators", 100, 500),
        "max_depth":        trial.suggest_int("max_depth", 5, 50),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        "max_features":     trial.suggest_categorical("max_features", ["sqrt", "log2"]),
        "class_weight":     "balanced",
        "n_jobs":           -1,
        "random_state":     random_state,
    }
    model = RandomForestClassifier(**params)
    scores = cross_val_score(model, X_train, y_train,
                             cv=cv, scoring="f1_macro", n_jobs=-1)
    return scores.mean()


def _et_objective(trial, X_train, y_train, cv, random_state):
    params = {
        "n_estimators":     trial.suggest_int("n_estimators", 100, 500),
        "max_depth":        trial.suggest_int("max_depth", 5, 50),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        "max_features":     trial.suggest_categorical("max_features", ["sqrt", "log2"]),
        "class_weight":     "balanced",
        "n_jobs":           -1,
        "random_state":     random_state,
    }
    model = ExtraTreesClassifier(**params)
    scores = cross_val_score(model, X_train, y_train,
                             cv=cv, scoring="f1_macro", n_jobs=-1)
    return scores.mean()


def _xgb_objective(trial, X_train, y_train, cv, random_state):
    xgb_params = {
        "n_estimators":      trial.suggest_categorical("n_estimators", [200, 300, 500]),
        "max_depth":         trial.suggest_int("max_depth", 4, 8),
        "learning_rate":     trial.suggest_float("learning_rate", 0.05, 0.2, log=True),
        "subsample":         trial.suggest_float("subsample", 0.7, 1.0),
        "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.7, 1.0),
        "eval_metric":       "mlogloss",
        "random_state":      random_state,
    }
    model = XGBWrapper(**xgb_params)
    scores = cross_val_score(model, X_train, y_train,
                             cv=cv, scoring="f1_macro", n_jobs=-1)
    return scores.mean()


_OBJECTIVES = {
    "random_forest": _rf_objective,
    "extra_trees":   _et_objective,
    "xgboost":       _xgb_objective,
}


def tune_model(
    model_name: str,
    X_train,
    y_train,
    n_trials: int = 50,
    fast_mode: bool = False,
    random_state: int = 42,
    csv_path: str = None,
) -> tuple:
    """
    Run Optuna TPE study for model_name.

    Returns: (best_estimator, best_params, best_cv_f1, trials_df)
    - best_estimator is ALREADY refit on the full X_train with best_params
    - trials_df is a DataFrame of all trial results (save to CSV for analysis)
    """
    if model_name not in _OBJECTIVES:
        raise ValueError(f"No Optuna objective for: {model_name}")

    actual_trials = max(5, n_trials // 5) if fast_mode else n_trials
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)

    objective_fn = _OBJECTIVES[model_name]

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=random_state),
        pruner=optuna.pruners.MedianPruner(),   # stops bad trials early
        study_name=model_name,
    )
    study.optimize(
        lambda trial: objective_fn(trial, X_train, y_train, cv, random_state),
        n_trials=actual_trials,
        show_progress_bar=True,
    )

    best_params   = study.best_params
    best_cv_f1    = study.best_value
    trials_df     = study.trials_dataframe()

    if csv_path:
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        trials_df.to_csv(csv_path, index=False)
        logger.info("Optuna trials saved: %s (%d rows)", csv_path, len(trials_df))

    logger.info("Best CV macro F1 (%s): %.4f", model_name, best_cv_f1)
    logger.info("Best params: %s", best_params)

    # Refit on FULL X_train with best params
    best_estimator = _refit(model_name, best_params, X_train, y_train, random_state)
    return best_estimator, best_params, best_cv_f1, trials_df


def _refit(model_name, params, X_train, y_train, random_state):
    if model_name == "random_forest":
        m = RandomForestClassifier(**params, class_weight="balanced",
                                   n_jobs=-1, random_state=random_state)
    elif model_name == "extra_trees":
        m = ExtraTreesClassifier(**params, class_weight="balanced",
                                 n_jobs=-1, random_state=random_state)
    elif model_name == "xgboost":
        m = XGBWrapper(**params)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    m.fit(X_train, y_train)
    return m
