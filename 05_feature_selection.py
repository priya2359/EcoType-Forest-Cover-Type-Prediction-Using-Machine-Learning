# %%  [markdown]
# # 07 â€” Feature Selection
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
# **Scope:** Fit RandomForest on SMOTE-augmented train data to rank feature importances. Drop features below threshold. Save exact column artifacts for inference alignment.
#
# | Item | Detail |
# |---|---|
# | Input | `X_train_smote.csv` / `y_train_smote.csv` (SMOTE-augmented, 59 features) |
# | Method | RandomForestClassifier feature importances |
# | Threshold | `feature_importance_threshold: 0.0005` (from `feature_config.yaml`) |
# | Expected output | 50â€“57 features (most sparse soil cols dropped) |
# | Key artifacts | `artifacts/feature_columns.txt` Â· `artifacts/quant_feature_columns.txt` |
#
# > **Golden Rule 2:** fit RF on train only. Test data is only transformed, never used to fit.
# >
# > **FAST_MODE:** Set `True` for quick exploration (fewer trees). Set `False` for final run.

print(f"PROJECT_ROOT: {PROJECT_ROOT}")

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yaml
import os

from src.feature_selector import (
    select_by_importance,
    apply_selection,
    save_feature_columns,
    save_quant_feature_columns,
)

# %%
FAST_MODE = True  # set False for final run (more trees, slower but stable importances)

with open('../configs/feature_config.yaml') as f:
    config = yaml.safe_load(f)

THRESHOLD    = config['feature_importance_threshold']  # 0.0005
RANDOM_STATE = config['random_state']                  # 42
N_ESTIMATORS = 50 if FAST_MODE else 200

print(f"FAST_MODE: {FAST_MODE}  |  n_estimators: {N_ESTIMATORS}  |  threshold: {THRESHOLD}")

# %%
X_train = pd.read_csv('../data/processed/X_train_smote.csv')
y_train = pd.read_csv('../data/processed/y_train_smote.csv').squeeze()
X_test  = pd.read_csv('../data/processed/X_test_fe.csv')
y_test  = pd.read_csv('../data/processed/y_test.csv').squeeze()

print(f"X_train (post-SMOTE): {X_train.shape}")
print(f"X_test  (original):   {X_test.shape}")
assert X_train.shape[1] == X_test.shape[1], "Train/test column count mismatch"
assert list(X_train.columns) == list(X_test.columns), "Column order mismatch between train and test"

# %%  [markdown]
# ## 1. Fit Feature Importance (Train Only)

# %%
selected_cols, importances = select_by_importance(
    X_train, y_train,
    threshold=THRESHOLD,
    n_estimators=N_ESTIMATORS,
    random_state=RANDOM_STATE,
)

dropped_cols = [c for c in X_train.columns if c not in selected_cols]
print(f"\nStarted with: {X_train.shape[1]} features")
print(f"Selected:     {len(selected_cols)}")
print(f"Dropped:      {len(dropped_cols)}")
print(f"\nDropped features:")
for c in sorted(dropped_cols):
    print(f"  {c}  (importance={importances[c]:.6f})")

# %%
# Feature importance bar chart -- top 30 for readability
top_n = 30
imp_sorted = importances.sort_values(ascending=False)
top_imp = imp_sorted.head(top_n)

fig, axes = plt.subplots(1, 2, figsize=(18, 6))

# Left: top 30
colors = ['#2196F3' if c in selected_cols else '#EF9A9A' for c in top_imp.index]
axes[0].barh(top_imp.index[::-1], top_imp.values[::-1], color=colors[::-1])
axes[0].axvline(THRESHOLD, color='red', linestyle='--', linewidth=1.2,
                label=f'Threshold ({THRESHOLD})')
axes[0].set_title(f'Top {top_n} Feature Importances')
axes[0].set_xlabel('Importance')
axes[0].legend()

# Right: all features sorted, coloured by keep/drop
all_colors = ['#2196F3' if c in selected_cols else '#EF9A9A' for c in imp_sorted.index]
axes[1].bar(range(len(imp_sorted)), imp_sorted.values, color=all_colors, edgecolor='none')
axes[1].axhline(THRESHOLD, color='red', linestyle='--', linewidth=1.2,
                label=f'Threshold ({THRESHOLD})')
axes[1].set_title('All Features â€” Keep (blue) vs Drop (red)')
axes[1].set_xlabel('Feature rank')
axes[1].set_ylabel('Importance')
axes[1].legend()

plt.tight_layout()
os.makedirs('../reports/figures/eda', exist_ok=True)
plt.savefig('../reports/figures/eda/08_feature_importance.png', dpi=150, bbox_inches='tight')
plt.show()

# %%  [markdown]
# ## 2. Apply Selection to Train and Test

# %%
X_train_sel = apply_selection(X_train, selected_cols)
X_test_sel  = apply_selection(X_test,  selected_cols)

print(f"X_train_sel: {X_train_sel.shape}")
print(f"X_test_sel:  {X_test_sel.shape}")
assert list(X_train_sel.columns) == list(X_test_sel.columns), "Column mismatch after selection"
assert X_train_sel.isnull().sum().sum() == 0
assert X_test_sel.isnull().sum().sum()  == 0
print("Assertions passed.")

# %%  [markdown]
# ## 3. Save Artifacts
#
# `artifacts/` paths match `predictor.py` defaults â€” do not change.

# %%
os.makedirs('../artifacts', exist_ok=True)

# Derive quant cols that survived selection (for StandardScaler at inference)
ohe_cols = set(
    [c for c in X_train_sel.columns if c.startswith('Wilderness_Area_')]
    + [c for c in X_train_sel.columns if c.startswith('Soil_Type_')]
)
quant_cols_selected = [c for c in X_train_sel.columns if c not in ohe_cols]

save_feature_columns(X_train_sel, '../artifacts/feature_columns.txt')
save_quant_feature_columns(quant_cols_selected, '../artifacts/quant_feature_columns.txt')

print(f"\nTotal selected features:   {len(selected_cols)}")
print(f"Quant cols (for scaler):   {len(quant_cols_selected)}")
print(f"OHE cols (no scaling):     {len(ohe_cols)}")

# %%
os.makedirs('../data/processed', exist_ok=True)

X_train_sel.reset_index(drop=True).to_csv('../data/processed/X_train_selected.csv', index=False)
X_test_sel.reset_index(drop=True).to_csv( '../data/processed/X_test_selected.csv',  index=False)

print(f"Saved X_train_selected.csv: {X_train_sel.shape}")
print(f"Saved X_test_selected.csv:  {X_test_sel.shape}")
print("y_train_smote.csv and y_test.csv unchanged -- nb08 loads them as-is.")

# %%
from src.feature_selector import load_feature_columns

feat_cols  = load_feature_columns('../artifacts/feature_columns.txt')
quant_cols = load_feature_columns('../artifacts/quant_feature_columns.txt')

# Expected quant cols (all 16 must survive selection -- they are high-importance by design)
EXPECTED_QUANT = [
    'Elevation', 'Slope',
    'Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology',
    'Horizontal_Distance_To_Roadways',
    'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm',
    'Horizontal_Distance_To_Fire_Points',
    'Aspect_sin', 'Aspect_cos',
    'Hydro_Distance_Combined', 'Hydro_Elev_interaction', 'Hillshade_mean',
    'Elevation_Slope_interaction', 'Distance_Road_Fire_Ratio',
]

wld_in_sel  = [c for c in feat_cols if c.startswith('Wilderness_Area_')]
soil_in_sel = [c for c in feat_cols if c.startswith('Soil_Type_')]

# 1. feature_columns.txt reloads and matches X_train_sel
assert feat_cols == list(X_train_sel.columns), "feature_columns.txt mismatch"

# 2. quant_feature_columns.txt has exactly 16 entries
assert len(quant_cols) == 16, f"Expected 16 quant cols, got {len(quant_cols)}: {quant_cols}"

# 3. All 16 expected quant cols survived selection
missing_quant = [c for c in EXPECTED_QUANT if c not in feat_cols]
assert not missing_quant, f"Quant cols dropped by selector (unexpected): {missing_quant}"

# 4. Wilderness cols == 4 (all 4 areas must be present)
assert len(wld_in_sel) == 4, f"Expected 4 wilderness cols, got {len(wld_in_sel)}"

# 5. Soil cols >= 20 (low-importance sparse cols dropped, but meaningful ones kept)
assert len(soil_in_sel) >= 20, f"Suspiciously few soil cols: {len(soil_in_sel)} -- check threshold"

print("All 5 assertions passed.")
print(f"\n  feature_columns.txt:       {len(feat_cols)} total columns")
print(f"  quant_feature_columns.txt: {len(quant_cols)} quant columns  (expected 16)")
print(f"  Wilderness cols:           {len(wld_in_sel)}  (expected 4)")
print(f"  Soil cols retained:        {len(soil_in_sel)} of 39")
print(f"  Soil cols dropped:         {39 - len(soil_in_sel)}")

# %%  [markdown]
# ## Summary
#
# | Item | Value |
# |---|---|
# | Input features | 59 |
# | Selected features | see output above |
# | Dropped features | mostly sparse Soil_Type cols |
# | Threshold | 0.0005 (from `feature_config.yaml`) |
# | `artifacts/feature_columns.txt` | exact inference column order |
# | `artifacts/quant_feature_columns.txt` | quant cols for StandardScaler |
#
# **Saved:**
# - `artifacts/feature_columns.txt` Â· `artifacts/quant_feature_columns.txt`
# - `data/processed/X_train_selected.csv` Â· `X_test_selected.csv`
#
# **Next step -> 08 Model Building**
