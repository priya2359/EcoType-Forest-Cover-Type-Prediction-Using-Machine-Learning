# %%  [markdown]
# # 08 -- Model Building + Hyperparameter Tuning
# %%
import sys
import os
from pathlib import Path

try:
    _NB = Path(__file__).resolve()
    PROJECT_ROOT = _NB.parent.parent
    os.chdir(_NB.parent)   # so ../data/ ../configs/ etc. resolve correctly
except NameError:
    PROJECT_ROOT = Path().resolve().parent   # Jupyter: cwd is notebooks/

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

#
# Trains all 6 models in one step. LR, DT, KNN use default params.
# RF, ET, XGB get Optuna TPE tuning before final refit.
#
# | Model              | Role            | Tuning         |
# |--------------------|-----------------|----------------|
# | Logistic Regression| Baseline        | None           |
# | Decision Tree      | Interpretable   | None           |
# | KNN                | GUVI required   | None (eval only)|
# | Random Forest      | Core ensemble   | Optuna 50 trials|
# | Extra Trees        | Bonus ensemble  | Optuna 50 trials|
# | XGBoost            | Primary candidate| Optuna 20 trials|
#
# Golden Rule 6: Set FAST_MODE = False for final models.
# Golden Rule 7: Each model saved immediately after training.
# Golden Rule 5: Macro F1 is primary metric.

print(f"PROJECT_ROOT: {PROJECT_ROOT}")

# %%
# Golden Rule 4: XGBWrapper BEFORE any joblib.load()
from src.xgb_wrapper import XGBWrapper  # noqa

import joblib
import pandas as pd
import numpy as np
import yaml
import os
import mlflow

from src.preprocessing import fit_scaler, transform_scaler, save_scaler
from src.model_trainer import get_model, train_model, log_run_to_mlflow, save_model
from src.model_evaluator import evaluate, plot_confusion_matrix, build_comparison_df, plot_model_comparison
from src.feature_selector import load_feature_columns
from src.hyperparameter_tuner import tune_model

# %%
FAST_MODE = False
with open('../configs/feature_config.yaml') as f:
    feature_config = yaml.safe_load(f)
with open('../configs/model_config.yaml') as f:
    model_config = yaml.safe_load(f)

class_map    = feature_config['class_map']
RANDOM_STATE = feature_config['random_state']

# Models that get Optuna tuning (others use config defaults)
TUNE_MODELS   = {'random_forest', 'extra_trees', 'xgboost'}
OPTUNA_TRIALS = {'random_forest': 50, 'extra_trees': 50, 'xgboost': 20}

# FAST_MODE: fewer estimators for non-tuned tree models, fewer Optuna trials
if FAST_MODE:
    model_config['models']['random_forest']['n_estimators'] = 50
    model_config['models']['extra_trees']['n_estimators']   = 50
    model_config['models']['xgboost']['n_estimators']       = 100
    print("FAST_MODE=True -- reduced estimators + Optuna trials")

mlflow.set_tracking_uri(model_config['mlflow']['tracking_uri'])
mlflow.set_experiment(model_config['mlflow']['experiment_name'])
print(f"MLflow experiment: {model_config['mlflow']['experiment_name']}")

# %%
X_train    = pd.read_csv('../data/processed/X_train_selected.csv')
y_train    = pd.read_csv('../data/processed/y_train_smote.csv').squeeze()
X_test     = pd.read_csv('../data/processed/X_test_selected.csv')
y_test     = pd.read_csv('../data/processed/y_test.csv').squeeze()
quant_cols = load_feature_columns('../artifacts/quant_feature_columns.txt')

assert X_train.shape[1] == X_test.shape[1], "Train/test column mismatch"
assert list(X_train.columns) == list(X_test.columns), "Column order mismatch"
assert all(c in X_train.columns for c in quant_cols), "Quant col missing"

print(f"X_train: {X_train.shape}  |  y_train: {y_train.shape}")
print(f"X_test:  {X_test.shape}   |  y_test:  {y_test.shape}")

# %%  [markdown]
# ## 1. StandardScaler -- Fit on Train, Transform Both
# Only quantitative columns scaled. OHE binary columns left as-is.

# %%
scaler = fit_scaler(X_train[quant_cols])

X_train_scaled = X_train.copy()
X_train_scaled[quant_cols] = transform_scaler(scaler, X_train, quant_cols).values

X_test_scaled = X_test.copy()
X_test_scaled[quant_cols] = transform_scaler(scaler, X_test, quant_cols).values

os.makedirs('../models/scalers', exist_ok=True)
save_scaler(scaler, '../models/scalers/standard_scaler.pkl')
print(f"Scaler saved. Quant cols scaled: {len(quant_cols)}")

# %%  [markdown]
# ## 2. Train All 6 Models (RF / ET / XGB with Optuna tuning)

# %%
MODEL_ORDER = [
    'logistic_regression',   # baseline -- no tuning
    'decision_tree',         # interpretable -- no tuning
    'random_forest',         # Optuna TPE 50 trials
    'extra_trees',           # Optuna TPE 50 trials
    'xgboost',               # Optuna TPE 20 trials
    'knn',                   # LAST: ball_tree predict is slow (30-120s)
]

os.makedirs('../models/trained', exist_ok=True)
os.makedirs('../models/best_model', exist_ok=True)
os.makedirs('../reports/figures/model', exist_ok=True)
os.makedirs('../reports/optuna', exist_ok=True)

results = {}

for name in MODEL_ORDER:
    print(f"\n{'='*55}\n{name.upper()}\n{'='*55}")

    if name in TUNE_MODELS:
        # Optuna TPE: tune on X_train, refit best params on full X_train
        n_trials = OPTUNA_TRIALS[name]
        print(f"Optuna tuning: {n_trials} trials ({'fast' if FAST_MODE else 'full'})")
        model, best_params, cv_f1, trials_df = tune_model(
            model_name=name,
            X_train=X_train_scaled,
            y_train=y_train,
            n_trials=n_trials,
            fast_mode=FAST_MODE,
            random_state=RANDOM_STATE,
            csv_path=f'../reports/optuna/{name}_trials.csv',
        )
        print(f"Best CV macro F1: {cv_f1:.4f}  |  trials: {len(trials_df)}")
        log_params = best_params
    else:
        # Default config params -- no tuning
        model = get_model(name, model_config)
        model = train_model(model, X_train_scaled, y_train)
        log_params = model_config['models'].get(name, {})
        cv_f1 = None

    # Golden Rule 7: save immediately after training
    save_model(model, f'../models/trained/{name}.pkl')

    metrics = evaluate(model, X_test_scaled, y_test, class_names=class_map)
    results[name] = metrics

    cm_path = f'../reports/figures/model/cm_{name}.png'
    plot_confusion_matrix(
        metrics['confusion_matrix'],
        class_names=[class_map[i] for i in sorted(class_map)],
        title=f'Confusion Matrix -- {name}',
        save_path=cm_path,
    )

    run_id = log_run_to_mlflow(
        model_name=name,
        model=model,
        metrics=metrics,
        params=log_params,
        cm_save_path=cm_path,
        class_map=class_map,
    )

    cv_str = f"  CV F1={cv_f1:.4f}" if cv_f1 else ""
    print(f"  Test macro F1={metrics['macro_f1']:.4f}  accuracy={metrics['accuracy']:.4f}{cv_str}")

# %%  [markdown]
# ## 3. Model Comparison

# %%
comparison_df = build_comparison_df(results)
print("Final comparison (sorted by macro F1):")
print(comparison_df[['model', 'macro_f1', 'weighted_f1', 'accuracy']].to_string(index=False))

plot_model_comparison(
    comparison_df,
    metric='macro_f1',
    save_path='../reports/figures/model/model_comparison_bar.png',
)

# %%  [markdown]
# ## 4. Save Best Model

# %%
best_name  = comparison_df.iloc[0]['model']
best_f1    = comparison_df.iloc[0]['macro_f1']
best_model = joblib.load(f'../models/trained/{best_name}.pkl')

save_model(best_model, '../models/best_model/ecotype_best_model.pkl')
print(f"Best model: {best_name}  (macro F1={best_f1:.4f})")
print(f"Saved to:   models/best_model/ecotype_best_model.pkl")

# %%
print(f"\nClassification Report -- {best_name}:")
cr = results[best_name]['classification_report']
for cls_name, row in cr.items():
    if isinstance(row, dict):
        print(f"  {cls_name:<22} precision={row['precision']:.4f}  recall={row['recall']:.4f}  f1={row['f1-score']:.4f}")

# %%  [markdown]
# ## Summary
#
# | Artifact | Path |
# |---|---|
# | Scaler | `models/scalers/standard_scaler.pkl` |
# | Trained models | `models/trained/{name}.pkl` (6 files) |
# | Best model | `models/best_model/ecotype_best_model.pkl` |
# | Optuna trials | `reports/optuna/{rf,et,xgboost}_trials.csv` |
# | Confusion matrices | `reports/figures/model/cm_{name}.png` |
# | MLflow runs | `mlruns/` -- view with `mlflow ui` |
#
# **Next step -> 10 Final Evaluation**
