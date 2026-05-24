# %%  [markdown]
# # 03 â€” Data Cleaning
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
# **Scope:** Encode target, fix edge cases, inspect (don't blindly remove) outliers, stratified 80/20 split, save `train_clean.csv` + `test_clean.csv`.
#
# | Task | Detail |
# |---|---|
# | Cover_Type encoding | String labels â†’ int 1â€“7 via `feature_config.yaml` class_map |
# | Aspect edge-case fix | Recode 360Â° â†’ 0Â° only (sin/cos decomposition is step 04 FE) |
# | Outlier inspection | Boxplots by Cover_Type â€” keep ecological signals, only flag impossible values |
# | Train/test split | Stratified 80/20 â€” `random_state` and `test_size` from config |
# | Outputs | `data/processed/train_clean.csv` Â· `data/processed/test_clean.csv` |

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
import os
from sklearn.model_selection import train_test_split

# %%
df = pd.read_csv(r"../data/raw/cover_type (1).csv")

with open("../configs/feature_config.yaml") as f:
    config = yaml.safe_load(f)

print(f"Loaded: {df.shape}")
df.head(3)

# %%  [markdown]
# ## 1. Cover_Type Encoding
#
# The raw CSV stores Cover_Type as string names (e.g. `'Lodgepole Pine'`). We reverse the `class_map` from `feature_config.yaml` to encode them as integers 1â€“7.

# %%
class_map = config["class_map"]            # {1: 'Spruce/Fir', ...}
label_to_int = {v: k for k, v in class_map.items()}

unseen = set(df['Cover_Type'].unique()) - set(label_to_int.keys())
if unseen:
    raise ValueError(f"Cover_Type labels not in class_map: {unseen}")

df['Cover_Type'] = df['Cover_Type'].map(label_to_int)

print("Encoding map:", label_to_int)
print("\nCover_Type distribution after encoding:")
print(df['Cover_Type'].value_counts().sort_index().to_string())

# %%  [markdown]
# ## 2. Aspect Edge-Case Fix â€” 360Â° â†’ 0Â°
#
# Aspect 360Â° is identical to 0Â° (north-facing). We recode it to eliminate the artificial boundary.
#
# > **Note:** The full circular treatment of Aspect (sin/cos decomposition) belongs in **step 04 Feature Engineering** â€” `Aspect_sin` and `Aspect_cos` will replace raw `Aspect` there.

# %%
n_360 = (df['Aspect'] == 360).sum()
df['Aspect'] = df['Aspect'].replace(360, 0)
print(f"Aspect==360 recoded: {n_360} rows")
print(f"Aspect range: [{df['Aspect'].min()}, {df['Aspect'].max()}]  (max should be â‰¤ 359)")

# %%  [markdown]
# ## 3. Outlier Inspection
#
# This dataset contains **ecological outliers** â€” extreme Elevation, steep Slope, distant fire points â€” that are **genuine geographic signals**, not data errors. For example:
# - Elevation â‰ˆ 3800 m â†’ almost certainly Krummholz (class 7)
# - Winsorizing these would **destroy the signal**
#
# Our tree models (DT, RF, ExtraTrees, XGB) are robust to extreme values. KNN and LR will receive scaled data via StandardScaler later, which handles range differences without clipping.
#
# **Strategy:** inspect only â€” remove rows only for physically impossible values.

# %%
# Validity check -- physically impossible values
validity_rules = {
    'Hillshade_9am':   (0, 255),
    'Hillshade_Noon':  (0, 255),
    'Hillshade_3pm':   (0, 255),
    'Slope':           (0, 90),
    'Aspect':          (0, 359),
    'Elevation':       (1500, 4500),
    'Wilderness_Area': (1, 4),
    'Soil_Type':       (1, 40),
    'Cover_Type':      (1, 7),
}

print("Validity audit (rows outside physically possible range):")
total_invalid = 0
for col, (lo, hi) in validity_rules.items():
    n = ((df[col] < lo) | (df[col] > hi)).sum()
    total_invalid += n
    status = "CLEAN" if n == 0 else f"[!!] {n:,} rows INVALID"
    print(f"  {col:<45} [{lo}, {hi}]  ->  {status}")

print(f"\nTotal invalid rows: {total_invalid}")
if total_invalid == 0:
    print("No data errors found -- no rows to drop.")

# %%
# Boxplots of each quant feature by Cover_Type
# Goal: confirm that 'outliers' are class-specific signals, not noise

# Derive from actual df columns -- config["quantitative_features"] already lists
# Aspect_sin/Aspect_cos (post-FE state), but here we still have raw Aspect.
quant_cols = [c for c in df.columns if c not in ('Wilderness_Area', 'Soil_Type', 'Cover_Type')]

df_plot = df.copy()
df_plot['Cover_Type_Name'] = df_plot['Cover_Type'].map(class_map)

n_cols = len(quant_cols)
fig, axes = plt.subplots(2, 5, figsize=(22, 9))
axes = axes.flatten()

for i, col in enumerate(quant_cols):
    df_plot.boxplot(
        column=col,
        by='Cover_Type_Name',
        ax=axes[i],
        sym='.',
        flierprops=dict(markersize=2, alpha=0.3)
    )
    axes[i].set_title(col, fontsize=8)
    axes[i].set_xlabel('')
    axes[i].tick_params(axis='x', rotation=45, labelsize=6)

plt.suptitle('Quant Feature Distributions by Cover Type\n(outlier dots = ecological signals, not noise)', y=1.01)
plt.tight_layout()
os.makedirs('../reports/figures/eda', exist_ok=True)
plt.savefig('../reports/figures/eda/01_outlier_boxplots.png', dpi=150, bbox_inches='tight')
plt.show()
print("Chart saved: reports/figures/eda/01_outlier_boxplots.png")
print("Observation: extreme Elevation values cluster in Krummholz -- class signal, not error.")

# %%  [markdown]
# ## 4. Stratified Train / Test Split
#
# - `test_size` and `random_state` are read from `feature_config.yaml` â€” single source of truth
# - Stratified on `Cover_Type` to preserve class proportions in both sets
# - **SMOTE will only be applied on `train_clean.csv` in step 06 â€” never on test data**

# %%
X = df.drop(columns=['Cover_Type'])
y = df['Cover_Type']

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=config['test_size'],
    random_state=config['random_state'],
    stratify=y
)

train_df = X_train.copy()
train_df['Cover_Type'] = y_train.values

test_df = X_test.copy()
test_df['Cover_Type'] = y_test.values

print(f"Train: {train_df.shape}  ({len(train_df)/len(df)*100:.0f}%)")
print(f"Test:  {test_df.shape}  ({len(test_df)/len(df)*100:.0f}%)")
print(f"\nClass distribution â€” train vs test (should be proportional):")
dist = pd.DataFrame({
    'Train': train_df['Cover_Type'].value_counts().sort_index(),
    'Test':  test_df['Cover_Type'].value_counts().sort_index()
})
dist['Train%'] = (dist['Train'] / dist['Train'].sum() * 100).round(2)
dist['Test%']  = (dist['Test']  / dist['Test'].sum()  * 100).round(2)
print(dist.to_string())

# %%  [markdown]
# ## 5. Save Cleaned Files

# %%
os.makedirs("../data/processed", exist_ok=True)

train_path = "../data/processed/train_clean.csv"
test_path  = "../data/processed/test_clean.csv"

train_df.to_csv(train_path, index=False)
test_df.to_csv(test_path, index=False)

print(f"Saved: {train_path}  ({train_df.shape[0]:,} rows Ã— {train_df.shape[1]} cols)")
print(f"Saved: {test_path}   ({test_df.shape[0]:,} rows Ã— {test_df.shape[1]} cols)")

# %%
# Verify saved files round-trip correctly
train_check = pd.read_csv(train_path)
test_check  = pd.read_csv(test_path)

assert train_check.shape == train_df.shape, "train_clean.csv shape mismatch"
assert test_check.shape  == test_df.shape,  "test_clean.csv shape mismatch"
assert train_check.isnull().sum().sum() == 0, "nulls in train_clean.csv"
assert test_check.isnull().sum().sum()  == 0, "nulls in test_clean.csv"
assert train_check['Cover_Type'].between(1, 7).all(), "Cover_Type out of range in train"
assert test_check['Cover_Type'].between(1, 7).all(),  "Cover_Type out of range in test"

print("All assertions passed.")
print(f"\ntrain_clean.csv dtypes:\n{train_check.dtypes.to_string()}")

# %%  [markdown]
# ## Summary
#
# | Step | Action | Result |
# |---|---|---|
# | Cover_Type encoding | String â†’ int 1â€“7 (class_map from config) | All 7 classes mapped, 0 unmapped |
# | Aspect fix | 360Â° â†’ 0Â° edge case | Max Aspect â‰¤ 359 |
# | Outlier inspection | Boxplots by class â€” no rows dropped | Extreme values are class signals |
# | Validity checks | Physically impossible ranges | 0 invalid rows |
# | Split | Stratified 80/20 | ~116K train Â· ~29K test |
#
# **Downstream notes:**
# - All notebooks from 04 onward load `train_clean.csv` and `test_clean.csv`
# - Aspect sin/cos decomposition â†’ step 04 Feature Engineering
# - SMOTE â†’ step 06, on training data only
#
# **Next step â†’ 04 Feature Engineering**
