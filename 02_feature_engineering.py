# %%  [markdown]
# # 04 â€” Feature Engineering
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
# **Scope:** Apply arithmetic transforms, Aspect circular decomposition, OHE for categoricals. No fitting on test data.
#
# | Task | Detail |
# |---|---|
# | `engineer_features()` | Adds Aspect_sin/cos (drops raw Aspect) + 5 domain features |
# | OHE Wilderness_Area | `fit` on train only â†’ `transform` train + test |
# | OHE Soil_Type | `fit` on train only â†’ `transform` train + test |
# | Artifacts saved | `ohe_wilderness.pkl`, `ohe_soil.pkl` only â€” `feature_columns.txt` saved in **nb07** after feature selection |
# | Data saved | `X_train_fe.csv`, `X_test_fe.csv`, `y_train.csv`, `y_test.csv` |

print(f"PROJECT_ROOT: {PROJECT_ROOT}")

# %%
import pandas as pd
import numpy as np
import yaml
import os
import joblib

from src.feature_engineering import engineer_features, ENGINEERED_COLS
from src.preprocessing import fit_ohe, transform_ohe, save_encoder

# %%
with open("../configs/feature_config.yaml") as f:
    config = yaml.safe_load(f)

train_df = pd.read_csv("../data/processed/train_clean.csv")
test_df  = pd.read_csv("../data/processed/test_clean.csv")

# Schema verification -- fail fast before any transforms
expected_cols = [
    'Elevation', 'Aspect', 'Slope',
    'Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology',
    'Horizontal_Distance_To_Roadways', 'Hillshade_9am', 'Hillshade_Noon',
    'Hillshade_3pm', 'Horizontal_Distance_To_Fire_Points',
    'Wilderness_Area', 'Soil_Type', 'Cover_Type'
]
for name, df in [('train', train_df), ('test', test_df)]:
    missing = set(expected_cols) - set(df.columns)
    if missing:
        raise ValueError(f"{name} missing columns: {missing}")

print(f"Train: {train_df.shape}  |  Test: {test_df.shape}")
print("Schema OK -- all expected columns present")

# %%  [markdown]
# ## 1. Separate Features and Target

# %%
X_train = train_df.drop(columns=['Cover_Type'])
y_train = train_df['Cover_Type']
X_test  = test_df.drop(columns=['Cover_Type'])
y_test  = test_df['Cover_Type']

print(f"X_train: {X_train.shape}  y_train: {y_train.shape}")
print(f"X_test:  {X_test.shape}   y_test:  {y_test.shape}")

# %%  [markdown]
# ## 2. Arithmetic Feature Engineering
#
# Pure arithmetic â€” no fitting, safe to apply to both train and test identically.
#
# - `Aspect_sin` / `Aspect_cos` â€” circular decomposition, raw `Aspect` dropped
# - `Hydro_Distance_Combined` â€” Euclidean distance to nearest water
# - `Hydro_Elev_interaction` â€” absolute elevation of water source
# - `Hillshade_mean` â€” average daily solar exposure
# - `Elevation_Slope_interaction` â€” high elev + steep slope predicts Krummholz
# - `Distance_Road_Fire_Ratio` â€” relative accessibility vs fire proximity

# %%
X_train = engineer_features(X_train)
X_test  = engineer_features(X_test)

print(f"After FE -- X_train: {X_train.shape}  |  X_test: {X_test.shape}")
print(f"Columns added: {ENGINEERED_COLS}")
print(f"Raw Aspect dropped: {'Aspect' not in X_train.columns}")
X_train.head(2)

# %%  [markdown]
# ## 3. One-Hot Encoding
#
# **Golden Rule:** `fit` only on train â€” `transform` on both. `get_feature_names_out()` used internally, never hardcoded.

# %%
# Wilderness_Area (4 categories)
ohe_wilderness = fit_ohe(X_train['Wilderness_Area'], 'Wilderness_Area')
wld_train = transform_ohe(ohe_wilderness, X_train['Wilderness_Area'], 'Wilderness_Area')
wld_test  = transform_ohe(ohe_wilderness, X_test['Wilderness_Area'],  'Wilderness_Area')

print(f"Wilderness_Area OHE: {wld_train.shape[1]} columns")
print(wld_train.columns.tolist())

# %%
# Soil_Type (up to 40 categories -- fit on train to avoid test leakage)
ohe_soil = fit_ohe(X_train['Soil_Type'], 'Soil_Type')
soil_train = transform_ohe(ohe_soil, X_train['Soil_Type'], 'Soil_Type')
soil_test  = transform_ohe(ohe_soil, X_test['Soil_Type'],  'Soil_Type')

n_soil_cols = soil_train.shape[1]
print(f"Soil_Type OHE: {n_soil_cols} columns (categories seen in train)")
print(f"Total features after nb04: 16 quant + 4 wilderness + {n_soil_cols} soil = {16 + 4 + n_soil_cols}")
print(soil_train.columns.tolist())

# %%  [markdown]
# ## 4. Assemble Final Feature Matrices

# %%
# Explicit index alignment before concat -- guards against any encoder refactor
# that might reset index internally (silent row misalignment shows up as NaN, not an error)
wld_train.index  = X_train.index
wld_test.index   = X_test.index
soil_train.index = X_train.index
soil_test.index  = X_test.index

X_train_fe = pd.concat(
    [X_train.drop(columns=['Wilderness_Area', 'Soil_Type']), wld_train, soil_train],
    axis=1
)
X_test_fe = pd.concat(
    [X_test.drop(columns=['Wilderness_Area', 'Soil_Type']), wld_test, soil_test],
    axis=1
)

print(f"X_train_fe: {X_train_fe.shape}")
print(f"X_test_fe:  {X_test_fe.shape}")
assert X_train_fe.shape[1] == X_test_fe.shape[1], "Column count mismatch between train and test"
assert X_train_fe.isnull().sum().sum() == 0, "NaN in X_train_fe -- index misalignment?"
assert X_test_fe.isnull().sum().sum()  == 0, "NaN in X_test_fe -- index misalignment?"
print("Assertions passed.")

# %%  [markdown]
# ## 5. Save Artifacts

# %%
os.makedirs("../models/encoders", exist_ok=True)
save_encoder(ohe_wilderness, "../models/encoders/wilderness_ohe.pkl")
save_encoder(ohe_soil,       "../models/encoders/soil_ohe.pkl")

# %%
# Derive column groups for verification only.
# feature_columns.txt and quant_feature_columns.txt are saved in nb07 AFTER feature selection.
# Saving them here would capture all ~59 pre-selection columns; nb07 drops low-importance
# soil type cols, so any pre-selection file would misalign with the trained model at inference.
ohe_cols = set(wld_train.columns.tolist() + soil_train.columns.tolist())
quant_feature_cols = [c for c in X_train_fe.columns if c not in ohe_cols]

print(f"Total columns:  {X_train_fe.shape[1]}")
print(f"Quant cols:     {len(quant_feature_cols)}  (expected 16)")
print(f"OHE cols:       {len(ohe_cols)}")
assert len(quant_feature_cols) == 16, f"Expected 16 quant cols, got {len(quant_feature_cols)} -- check if raw Aspect was dropped"

# %%
os.makedirs("../data/processed", exist_ok=True)

X_train_fe.reset_index(drop=True).to_csv("../data/processed/X_train_fe.csv", index=False)
X_test_fe.reset_index(drop=True).to_csv( "../data/processed/X_test_fe.csv",  index=False)
y_train.reset_index(drop=True).to_csv(   "../data/processed/y_train.csv",     index=False)
y_test.reset_index(drop=True).to_csv(    "../data/processed/y_test.csv",      index=False)

print(f"Saved X_train_fe: {X_train_fe.shape}")
print(f"Saved X_test_fe:  {X_test_fe.shape}")
print(f"Saved y_train:    {y_train.shape}")
print(f"Saved y_test:     {y_test.shape}")

# %%
# Round-trip verification
X_tr_check = pd.read_csv("../data/processed/X_train_fe.csv")
X_te_check = pd.read_csv("../data/processed/X_test_fe.csv")
y_tr_check = pd.read_csv("../data/processed/y_train.csv")
y_te_check = pd.read_csv("../data/processed/y_test.csv")

assert X_tr_check.shape == X_train_fe.shape
assert X_te_check.shape == X_test_fe.shape
assert list(X_tr_check.columns) == list(X_train_fe.columns), "Column order mismatch on reload"
assert y_tr_check.squeeze().between(1, 7).all(), "y_train values out of range"
assert y_te_check.squeeze().between(1, 7).all(), "y_test values out of range"

print("All round-trip assertions passed.")
print(f"\nFinal feature matrix shape: {X_tr_check.shape}")
print(f"Column breakdown:")
print(f"  Quantitative (to be scaled): {len(quant_feature_cols)}")
print(f"  OHE (binary, no scaling):    {len(ohe_cols)}")

# %%  [markdown]
# ## Summary
#
# | Step | Action | Result |
# |---|---|---|
# | `engineer_features()` | +7 cols (Aspect_sin/cos + 5 domain), raw Aspect dropped | net +6 columns |
# | OHE Wilderness_Area | fit on train | 4 binary cols |
# | OHE Soil_Type | fit on train | N binary cols (categories seen in train) |
# | Quant cols verified | 10 original - 1 Aspect + 2 sin/cos + 5 domain = **16** | assert passes |
#
# **Saved files (this notebook):**
# - `models/ohe_wilderness.pkl` Â· `models/ohe_soil.pkl`
# - `data/processed/X_train_fe.csv` Â· `X_test_fe.csv` Â· `y_train.csv` Â· `y_test.csv`
#
# **Deferred to nb07 (after feature selection):**
# - `models/feature_columns.txt` â€” exact final column order for inference
# - `models/quant_feature_columns.txt` â€” quant cols for StandardScaler
#
# **Next step -> 05 EDA**
