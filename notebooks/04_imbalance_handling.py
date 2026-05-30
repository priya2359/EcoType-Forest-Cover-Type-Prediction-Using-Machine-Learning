# %%  [markdown]
# # 06 â€” Imbalance Handling
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
# **Scope:** SMOTE on training data only. Explicit per-class target dict â€” never `'auto'` or `'minority'`.
#
# | Decision | Detail |
# |---|---|
# | SMOTE targets | Classes 3, 4, 6, 7 (1,728 rows each â€” below 2,000 threshold) |
# | Class 5 Aspen | 2,455 rows â€” above threshold, untouched |
# | Target count | 2,455 per minority class (matches class 5 naturally) |
# | sampling_strategy | Explicit dict `{3:2455, 4:2455, 6:2455, 7:2455}` â€” classes 1, 2, 5 untouched |
# | Test data | Never touched â€” SMOTE on train only (Golden Rule 1) |

print(f"PROJECT_ROOT: {PROJECT_ROOT}")

# %%
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import yaml
import os

from src.imbalance_handler import apply_smote, get_class_distribution

# %%
with open('../configs/feature_config.yaml') as f:
    config = yaml.safe_load(f)
class_map = config['class_map']

X_train = pd.read_csv('../data/processed/X_train_fe.csv')
y_train = pd.read_csv('../data/processed/y_train.csv').squeeze()

print(f"X_train: {X_train.shape}  |  y_train: {y_train.shape}")
print("\nPre-SMOTE class distribution:")
for cls, info in get_class_distribution(y_train).items():
    flag = ' <- SMOTE target' if info['count'] < config['smote_threshold'] else ''
    print(f"  {cls} {class_map[cls]:<22} {info['count']:>7,}  ({info['pct']:.2f}%){flag}")

# %%  [markdown]
# ## Apply SMOTE
#
# Explicit `sampling_strategy` dict ensures only classes 3, 4, 6, 7 are oversampled to 2,455.
# Classes 1, 2, 5 are not listed and remain untouched.

# %%
SMOTE_TARGET = 2455  # match class 5 Aspen naturally

sampling_strategy = {
    3: SMOTE_TARGET,  # Ponderosa Pine:    1728 -> 2455
    4: SMOTE_TARGET,  # Cottonwood/Willow: 1728 -> 2455
    6: SMOTE_TARGET,  # Douglas-fir:       1728 -> 2455
    7: SMOTE_TARGET,  # Krummholz:         1728 -> 2455
    # Classes 1, 2, 5 not listed -- left untouched
}

X_res, y_res = apply_smote(
    X_train, y_train,
    sampling_strategy=sampling_strategy,
    random_state=config['random_state'],
    k_neighbors=5,
)

print(f"Before SMOTE: {X_train.shape[0]:,} rows")
print(f"After SMOTE:  {X_res.shape[0]:,} rows  (+{X_res.shape[0]-X_train.shape[0]:,} synthetic)")

# %%
# Verify post-SMOTE distribution
print("Post-SMOTE class distribution:")
post_dist = get_class_distribution(y_res)
for cls, info in post_dist.items():
    pre = get_class_distribution(y_train)[cls]['count']
    added = info['count'] - pre
    tag = f" (+{added} synthetic)" if added > 0 else " (unchanged)"
    print(f"  {cls} {class_map[cls]:<22} {info['count']:>7,}  ({info['pct']:.2f}%){tag}")

# Assertions
assert y_res.value_counts()[3] == SMOTE_TARGET, f"Class 3 count wrong: {y_res.value_counts()[3]}"
assert y_res.value_counts()[4] == SMOTE_TARGET, f"Class 4 count wrong"
assert y_res.value_counts()[6] == SMOTE_TARGET, f"Class 6 count wrong"
assert y_res.value_counts()[7] == SMOTE_TARGET, f"Class 7 count wrong"
assert y_res.value_counts()[1] == y_train.value_counts()[1], "Class 1 changed -- should be untouched"
assert y_res.value_counts()[2] == y_train.value_counts()[2], "Class 2 changed -- should be untouched"
assert y_res.value_counts()[5] == y_train.value_counts()[5], "Class 5 changed -- should be untouched"
assert X_res.isnull().sum().sum() == 0, "Nulls in X_res after SMOTE"
print("\nAll assertions passed.")

# %%
# Post-SMOTE range validation — SMOTE can extrapolate beyond valid ranges on quant features
quant_bounds = {
    'Hillshade_9am': (0, 255), 'Hillshade_Noon': (0, 255), 'Hillshade_3pm': (0, 255),
    'Slope': (0, 90),
}
print("\nPost-SMOTE range validation:")
for col, (lo, hi) in quant_bounds.items():
    if col not in X_res.columns:
        continue
    n_oob = ((X_res[col] < lo) | (X_res[col] > hi)).sum()
    if n_oob > 0:
        print(f"  WARNING: {n_oob} synthetic rows have {col} outside [{lo},{hi}] -- clipping")
        X_res[col] = X_res[col].clip(lo, hi)
    else:
        print(f"  {col}: OK")
# OHE cols (Wilderness_Area_x, Soil_Type_x) will be continuous after SMOTE (e.g. 0.3, 0.7).
# Tree models are threshold-based -- handle this correctly.
# StandardScaler (nb06) scales these too; acceptable for RF/ET/XGB, slightly affects KNN distances.
print("Post-SMOTE validation complete.")

# %%
# Before vs after bar chart
pre_counts  = y_train.value_counts().sort_index()
post_counts = y_res.value_counts().sort_index()
names = [class_map[i] for i in pre_counts.index]
x = np.arange(len(names))
width = 0.4

fig, ax = plt.subplots(figsize=(12, 5))
ax.bar(x - width/2, pre_counts.values,  width, label='Before SMOTE', color='#90CAF9')
ax.bar(x + width/2, post_counts.values, width, label='After SMOTE',  color='#2196F3')
ax.axhline(SMOTE_TARGET, color='red', linestyle='--', linewidth=1, label=f'Target ({SMOTE_TARGET:,})')
ax.set_xticks(x)
ax.set_xticklabels(names, rotation=30, ha='right')
ax.set_title('Class Distribution Before vs After SMOTE')
ax.set_ylabel('Sample Count')
ax.legend()
plt.tight_layout()
os.makedirs('../reports/figures/eda', exist_ok=True)
plt.savefig('../reports/figures/eda/08_smote_distribution.png', dpi=150, bbox_inches='tight')
plt.close()

# %%  [markdown]
# ## Save Resampled Training Data

# %%
os.makedirs('../data/processed', exist_ok=True)

X_res.to_csv('../data/processed/X_train_smote.csv', index=False)
y_res.to_csv('../data/processed/y_train_smote.csv', index=False)

print(f"Saved X_train_smote.csv: {X_res.shape}")
print(f"Saved y_train_smote.csv: {y_res.shape}")
print("\nTest data untouched -- X_test_fe.csv and y_test.csv remain as-is.")

# %%
# Round-trip check
X_check = pd.read_csv('../data/processed/X_train_smote.csv')
y_check = pd.read_csv('../data/processed/y_train_smote.csv').squeeze()

assert X_check.shape == X_res.shape, "X shape mismatch on reload"
assert list(X_check.columns) == list(X_res.columns), "Column order mismatch"
assert X_check.isnull().sum().sum() == 0, "Nulls in saved file"
assert y_check.value_counts()[5] == 2455, "Class 5 count wrong after reload"

print(f"Round-trip OK -- {X_check.shape[0]:,} rows x {X_check.shape[1]} cols")

# %%  [markdown]
# ## Summary
#
# | Item | Value |
# |---|---|
# | SMOTE method | `sampling_strategy` dict â€” explicit per-class targets |
# | Classes oversampled | 3, 4, 6, 7 (1,728 â†’ 2,455 each) |
# | Classes untouched | 1, 2, 5 |
# | Synthetic rows added | 4 Ã— (2,455 - 1,728) = 2,908 |
# | Final train size | Original + 2,908 synthetic rows |
# | Test data | Untouched â€” `X_test_fe.csv` / `y_test.csv` unchanged |
#
# **Saved:** `data/processed/X_train_smote.csv` Â· `y_train_smote.csv`
#
# **Next step -> 07 Feature Selection**
