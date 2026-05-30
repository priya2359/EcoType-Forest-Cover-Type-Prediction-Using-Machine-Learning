# %%  [markdown]
# # 10 -- Final Evaluation
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
# Crown the winning tuned model, run full held-out test evaluation,
# register champion to MLflow model registry, save best_model pkl.
#
# Baselines (nb08 FAST_MODE=False): RF 0.9149 | ET 0.9062
# Set FAST_MODE = False before the final run.

print(f"PROJECT_ROOT: {PROJECT_ROOT}")

# %%
# Golden Rule 4: XGBWrapper before any joblib.load()
from src.xgb_wrapper import XGBWrapper  # noqa

import joblib
import pandas as pd
import numpy as np
import yaml
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from mlflow.models import infer_signature
from mlflow.exceptions import MlflowException

from src.model_trainer import save_model
from src.model_evaluator import evaluate, plot_confusion_matrix, build_comparison_df
from src.feature_selector import load_feature_columns
from src.preprocessing import transform_scaler

# %%
FAST_MODE = False  # Always False for final evaluation

with open('../configs/feature_config.yaml') as f:
    feature_config = yaml.safe_load(f)
with open('../configs/model_config.yaml') as f:
    model_config = yaml.safe_load(f)

class_map    = feature_config['class_map']
RANDOM_STATE = feature_config['random_state']

# Single tracking URI for runs AND registry -- never mix with sqlite
mlflow.set_tracking_uri(model_config['mlflow']['tracking_uri'])
client = MlflowClient(tracking_uri=model_config['mlflow']['tracking_uri'])
print(f"MLflow tracking URI: {model_config['mlflow']['tracking_uri']}")

# %%
X_test     = pd.read_csv('../data/processed/X_test_selected.csv')
y_test     = pd.read_csv('../data/processed/y_test.csv').squeeze()
quant_cols = load_feature_columns('../artifacts/quant_feature_columns.txt')

# Transform only -- scaler was fit on X_train in nb08, never refit
scaler = joblib.load('../models/scalers/standard_scaler.pkl')
X_test_sc = X_test.copy()
X_test_sc[quant_cols] = transform_scaler(scaler, X_test, quant_cols).values

print(f"X_test_sc: {X_test_sc.shape}  |  y_test: {y_test.shape}")

# %%  [markdown]
# ## 1. Load All Models and Evaluate

# %%
# Load all 6 baseline models + 2 tuned models
model_paths = {
    'logistic_regression': '../models/trained/logistic_regression.pkl',
    'decision_tree':       '../models/trained/decision_tree.pkl',
    'knn':                 '../models/trained/knn.pkl',
    'random_forest':       '../models/trained/random_forest.pkl',
    'extra_trees':         '../models/trained/extra_trees.pkl',
    'xgboost':             '../models/trained/xgboost.pkl',
}

# nb08 baselines (FAST_MODE=False) -- hardcoded for comparison table
BASELINES = {
    'logistic_regression': {'macro_f1': 0.6391, 'accuracy': 0.6705, 'weighted_f1': 0.7039},
    'decision_tree':       {'macro_f1': 0.8665, 'accuracy': 0.9323, 'weighted_f1': 0.9331},
    'knn':                 {'macro_f1': 0.8162, 'accuracy': 0.9048, 'weighted_f1': 0.9033},
    'random_forest':       {'macro_f1': 0.9149, 'accuracy': 0.9582, 'weighted_f1': 0.9577},
    'extra_trees':         {'macro_f1': 0.9062, 'accuracy': 0.9481, 'weighted_f1': 0.9472},
    'xgboost':             {'macro_f1': 0.9043, 'accuracy': 0.9424, 'weighted_f1': 0.9414},
}

tuned_results = {}
for name, path in model_paths.items():
    if not os.path.exists(path):
        print(f"SKIP (not found): {path}")
        continue
    model = joblib.load(path)
    metrics = evaluate(model, X_test_sc, y_test, class_names=class_map)
    tuned_results[name] = {'model': model, 'metrics': metrics}
    baseline_f1 = BASELINES.get(name, {}).get('macro_f1', 0)
    delta = metrics['macro_f1'] - baseline_f1
    print(f"  {name:<25} macro F1: {metrics['macro_f1']:.4f}  ({delta:+.4f} vs baseline)")

# %%  [markdown]
# ## 2. Crown the Winner

# %%
# Sort by macro F1 descending
sorted_models = sorted(tuned_results.items(), key=lambda x: -x[1]['metrics']['macro_f1'])

print("Final ranking (tuned models on test set):")
for rank, (name, result) in enumerate(sorted_models, 1):
    m = result['metrics']
    baseline_f1 = BASELINES.get(name, {}).get('macro_f1', 0)
    marker = ' <- CHAMPION' if rank == 1 else ''
    print(f"  {rank}. {name:<25} macro F1: {m['macro_f1']:.4f}  accuracy: {m['accuracy']:.4f}{marker}")

champion_name   = sorted_models[0][0]
champion_model  = sorted_models[0][1]['model']
champion_metrics = sorted_models[0][1]['metrics']

print(f"\nChampion: {champion_name}")
print(f"  macro F1:    {champion_metrics['macro_f1']:.4f}")
print(f"  accuracy:    {champion_metrics['accuracy']:.4f}")
print(f"  weighted F1: {champion_metrics['weighted_f1']:.4f}")

# %%
# Compute predictions ONCE — reused by per-wilderness F1 and calibration curves below
# X_test_sc is a DataFrame (created via X_test.copy() + column assignment)
from sklearn.metrics import f1_score
from sklearn.calibration import calibration_curve
from pathlib import Path

X_test_r = X_test.reset_index(drop=True)
y_test_r = y_test.reset_index(drop=True)
if hasattr(X_test_sc, 'reset_index'):
    X_test_sc_vals = X_test_sc.reset_index(drop=True).values
else:
    X_test_sc_vals = np.asarray(X_test_sc)
    assert len(X_test_sc_vals) == len(X_test_r), (
        f"Shape mismatch: X_test_sc has {len(X_test_sc_vals)} rows, "
        f"X_test has {len(X_test_r)} rows"
    )

y_pred_champ = champion_model.predict(X_test_sc_vals)
proba_matrix = champion_model.predict_proba(X_test_sc_vals)

# Per-wilderness-area macro F1 breakdown
wld_cols = [c for c in X_test_r.columns if c.startswith('Wilderness_Area_')]
if wld_cols:
    wilderness_ids = (
        X_test_r[wld_cols]
        .idxmax(axis=1)
        .str.extract(r'(\d+)$', expand=False)
        .astype(int)
    )
    print("\nPer-Wilderness-Area macro F1 (champion model):")
    for wid in [1, 2, 3, 4]:
        mask = (wilderness_ids == wid).values
        if mask.sum() == 0:
            print(f"  Area {wid}: no test samples — skipped")
            continue
        area_f1 = f1_score(
            y_test_r.values[mask],
            y_pred_champ[mask],
            average='macro',
            zero_division=0,
            labels=list(range(1, 8)),
        )
        print(f"  Area {wid}: macro F1 = {area_f1:.4f}  (n={mask.sum():,})")
else:
    print("Wilderness_Area OHE cols not in selected features — per-area breakdown skipped.")

# %%  [markdown]
# ## 3. Save Champion to best_model/

# %%
os.makedirs('../models/best_model', exist_ok=True)
save_model(champion_model, '../models/best_model/ecotype_best_model.pkl')
print(f"Saved champion ({champion_name}) to models/best_model/ecotype_best_model.pkl")

# %%  [markdown]
# ## 4. Full Classification Report + Confusion Matrix

# %%
print(f"Classification Report -- {champion_name}:")
cr = champion_metrics['classification_report']
print(f"  {'Class':<22} Precision  Recall  F1-Score  Support")
print(f"  {'-'*65}")
for cls_id, cls_name in class_map.items():
    if cls_name in cr and isinstance(cr[cls_name], dict):
        row = cr[cls_name]
        print(f"  {cls_name:<22} {row['precision']:.4f}    {row['recall']:.4f}  {row['f1-score']:.4f}    {int(row['support'])}")
print(f"\n  macro avg:             {cr['macro avg']['precision']:.4f}    {cr['macro avg']['recall']:.4f}  {cr['macro avg']['f1-score']:.4f}")
print(f"  weighted avg:          {cr['weighted avg']['precision']:.4f}    {cr['weighted avg']['recall']:.4f}  {cr['weighted avg']['f1-score']:.4f}")

os.makedirs('../reports/figures/model', exist_ok=True)
plot_confusion_matrix(
    champion_metrics['confusion_matrix'],
    class_names=[class_map[i] for i in sorted(class_map)],
    title=f'Confusion Matrix -- {champion_name} (Final Champion)',
    save_path='../reports/figures/model/cm_champion.png',
)

# %%
# Calibration curves — verify probability estimates are well-calibrated
sorted_classes = sorted(class_map.items())  # [(1,"Spruce/Fir"), ...] — guaranteed order

# Soft remap if model's internal class order differs from class_map (never hard assert in notebooks)
cal_proba = proba_matrix.copy()
if hasattr(champion_model, 'classes_'):
    model_cls = list(champion_model.classes_)
    expected  = [cid for cid, _ in sorted_classes]
    if model_cls != expected:
        print(f"WARNING: class order mismatch — model: {model_cls}, map: {expected}")
        try:
            col_order = [model_cls.index(cid) for cid in expected]
            cal_proba = cal_proba[:, col_order]
            print("  Remap successful.")
        except ValueError as e:
            print(f"  Remap failed ({e}) — skipping calibration curves.")
            cal_proba = None

if cal_proba is not None:
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    axes = axes.flatten()
    for i, (cls_id, cls_name) in enumerate(sorted_classes):
        y_binary = (y_test_r.values == cls_id).astype(int)
        if y_binary.sum() == 0 or y_binary.sum() == len(y_binary):
            axes[i].text(0.5, 0.5, 'Insufficient data', ha='center', va='center')
            axes[i].set_title(f"Class {cls_id}: {cls_name}", fontsize=9)
            continue
        fop, mpv = calibration_curve(y_binary, cal_proba[:, i], n_bins=10)
        axes[i].plot(mpv, fop, 's-', label=cls_name)
        axes[i].plot([0, 1], [0, 1], 'k--', label='Perfect')
        axes[i].set_title(f"Class {cls_id}: {cls_name}", fontsize=9)
        axes[i].set_xlabel('Mean predicted prob', fontsize=8)
        axes[i].set_ylabel('Fraction positives', fontsize=8)
        axes[i].legend(fontsize=7)
    axes[-1].set_visible(False)
    plt.suptitle(f'Calibration Curves — {champion_name}', fontsize=12)
    plt.tight_layout()
    cal_path = Path('../reports/figures/model/calibration_curves.png')
    cal_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(cal_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Calibration curves saved: {cal_path.resolve()}")

# %%  [markdown]
# ## 5. MLflow -- Log Champion + Register to Model Registry

# %%
mlflow.set_experiment(model_config['mlflow']['experiment_name'])

with mlflow.start_run(run_name=f'final_champion_{champion_name}') as run:
    # Log champion metrics
    mlflow.log_param('champion_model', champion_name)
    mlflow.log_metric('test_macro_f1',    champion_metrics['macro_f1'])
    mlflow.log_metric('test_accuracy',    champion_metrics['accuracy'])
    mlflow.log_metric('test_weighted_f1', champion_metrics['weighted_f1'])

    # Per-class F1
    for cls_id, cls_name in class_map.items():
        if cls_name in cr and isinstance(cr[cls_name], dict):
            mlflow.log_metric(f'f1_class_{cls_id}', cr[cls_name]['f1-score'])

    mlflow.log_artifact('../reports/figures/model/cm_champion.png')

    # Log model with signature -- use predict() output shape (n,) not predict_proba (n,7)
    # This matches what downstream users call: model.predict(X)
    sig = infer_signature(X_test_sc, champion_model.predict(X_test_sc))
    mlflow.sklearn.log_model(
        sk_model=champion_model,
        artifact_path='model',
        signature=sig,
        input_example=X_test_sc.iloc[:5],
    )
    champion_run_id = run.info.run_id

print(f"Logged to MLflow. run_id: {champion_run_id}")

# %%
# Register model -- safe for re-runs (try/except RESOURCE_ALREADY_EXISTS)
REGISTRY_NAME = 'ecotype_forest_cover'

try:
    client.create_registered_model(
        name=REGISTRY_NAME,
        description='EcoType forest cover type classifier. 7-class multiclass. Champion = highest macro F1.'
    )
    print(f"Created registered model: {REGISTRY_NAME}")
except MlflowException as e:
    if 'RESOURCE_ALREADY_EXISTS' in str(e):
        print(f"Registered model already exists: {REGISTRY_NAME}")
    else:
        raise

mv = client.create_model_version(
    name=REGISTRY_NAME,
    source=f'runs:/{champion_run_id}/model',
    run_id=champion_run_id,
    description=f'Champion: {champion_name}. macro F1={champion_metrics["macro_f1"]:.4f}.',
    tags={'model_name': champion_name},
)
client.set_registered_model_alias(name=REGISTRY_NAME, alias='champion', version=str(mv.version))
print(f"Registered: {REGISTRY_NAME} v{mv.version}  alias=champion")

# Verify alias is set
champ_mv = client.get_model_version_by_alias(REGISTRY_NAME, 'champion')
print(f"Champion alias verified: version {champ_mv.version}, model={champ_mv.tags.get('model_name')}")

# %%  [markdown]
# ## 6. Assertions

# %%
assert os.path.exists('../models/best_model/ecotype_best_model.pkl'), "best_model pkl missing"
assert champion_metrics['macro_f1'] > 0.91, f"macro F1 {champion_metrics['macro_f1']:.4f} < 0.91"
champ_check = client.get_model_version_by_alias(REGISTRY_NAME, 'champion')
assert champ_check is not None, "champion alias not set in registry"

print("All assertions passed.")

# %%  [markdown]
# ## Summary

# %%
print("=" * 70)
print("   SECTION 10 COMPLETE -- FINAL EVALUATION")
print("=" * 70)
print(f"\nChampion model: {champion_name}")
print(f"  macro F1:    {champion_metrics['macro_f1']:.4f}")
print(f"  accuracy:    {champion_metrics['accuracy']:.4f}")
print(f"  weighted F1: {champion_metrics['weighted_f1']:.4f}")
print(f"\nAll models ranked:")
for rank, (name, result) in enumerate(sorted_models, 1):
    m = result['metrics']
    print(f"  {rank}. {name:<25} {m['macro_f1']:.4f}")
print(f"\nSaved:  models/best_model/ecotype_best_model.pkl")
print(f"Registry: {REGISTRY_NAME} v{mv.version}  alias=champion")
print(f"\nTo load champion:")
print(f"  import mlflow.sklearn")
print(f"  model = mlflow.sklearn.load_model('models:/{REGISTRY_NAME}@champion')")
print(f"\nTo launch MLflow UI:")
print(f"  mlflow ui --backend-store-uri {model_config['mlflow']['tracking_uri']} --port 5000")
