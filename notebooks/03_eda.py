# %%  [markdown]
# # 05 â€” Exploratory Data Analysis
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
# **Scope:** Train data only â€” never peek at test. Understand class imbalance, feature distributions, correlations, and feature-class relationships to guide feature selection and modelling.
#
# | Section | Goal |
# |---|---|
# | Class distribution | Quantify imbalance â€” informs SMOTE threshold in nb06 |
# | Quant feature distributions | Spot skew, range, outlier patterns |
# | Correlation heatmap | Find multicollinearity candidates for nb07 |
# | Feature vs class | Which features best separate Cover_Type? |
# | Wilderness vs class | Is Wilderness_Area a strong discriminator? |
# | Engineered features | Do the 7 new features add separation? |
#
# All plots saved to `reports/figures/eda/`.

print(f"PROJECT_ROOT: {PROJECT_ROOT}")

# %%
import matplotlib
matplotlib.use('Agg')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import yaml
import os

sns.set_theme(style='whitegrid', palette='muted')
SAVE_DIR = '../reports/figures/eda'
os.makedirs(SAVE_DIR, exist_ok=True)

# %%
with open('../configs/feature_config.yaml') as f:
    config = yaml.safe_load(f)

class_map = config['class_map']  # {1: 'Spruce/Fir', ...}

X_train = pd.read_csv('../data/processed/X_train_fe.csv')
y_train = pd.read_csv('../data/processed/y_train.csv').squeeze()

print(f"X_train: {X_train.shape}  |  y_train: {y_train.shape}")

# Map numeric labels to names for all plots
y_names = y_train.map(class_map)

# Column groups
ohe_wld_cols  = [c for c in X_train.columns if c.startswith('Wilderness_Area_')]
ohe_soil_cols = [c for c in X_train.columns if c.startswith('Soil_Type_')]
quant_cols    = [c for c in X_train.columns if c not in ohe_wld_cols + ohe_soil_cols]

print(f"Quant: {len(quant_cols)}  |  Wilderness OHE: {len(ohe_wld_cols)}  |  Soil OHE: {len(ohe_soil_cols)}")

# %%  [markdown]
# ## 1. Class Distribution

# %%
counts = y_train.value_counts().sort_index()
names  = [class_map[i] for i in counts.index]

fig, ax = plt.subplots(figsize=(10, 4))
bars = ax.bar(names, counts.values,
              color=['#2196F3','#4CAF50','#FF9800','#9C27B0','#F44336','#00BCD4','#795548'])
ax.set_title('Cover Type Class Distribution (Train Set)', fontsize=13)
ax.set_ylabel('Sample Count')
ax.tick_params(axis='x', rotation=30)
for bar, val in zip(bars, counts.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 200,
            f'{val:,}', ha='center', va='bottom', fontsize=9)
plt.tight_layout()
plt.savefig(f'{SAVE_DIR}/01_class_distribution.png', dpi=150, bbox_inches='tight')
plt.close()

print(f"\nClass balance ratio (max/min): {counts.max()/counts.min():.1f}x")
print(f"Minority classes (< {config['smote_threshold']:,} samples):")
for i, cnt in counts.items():
    flag = ' <- SMOTE target' if cnt < config['smote_threshold'] else ''
    print(f"  {i} {class_map[i]:<20} {cnt:>7,}{flag}")

# %%  [markdown]
# ## 2. Quantitative Feature Distributions

# %%
fig, axes = plt.subplots(4, 4, figsize=(20, 16))
axes = axes.flatten()

for i, col in enumerate(quant_cols):
    axes[i].hist(X_train[col], bins=50, color='steelblue', edgecolor='none', alpha=0.8)
    axes[i].set_title(col, fontsize=8)
    axes[i].set_ylabel('Count', fontsize=7)
    skew = X_train[col].skew()
    axes[i].set_xlabel(f'skew={skew:.2f}', fontsize=7)

# Hide unused axes
for j in range(len(quant_cols), len(axes)):
    axes[j].set_visible(False)

plt.suptitle('Quantitative Feature Distributions (Train Set)', fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig(f'{SAVE_DIR}/02_quant_distributions.png', dpi=150, bbox_inches='tight')
plt.close()

# %%  [markdown]
# ## 3. Correlation Heatmap (Quantitative Features)

# %%
corr = X_train[quant_cols].corr()

fig, ax = plt.subplots(figsize=(14, 12))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(
    corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
    center=0, vmin=-1, vmax=1, linewidths=0.5,
    annot_kws={'size': 7}, ax=ax
)
ax.set_title('Quantitative Feature Correlation Matrix', fontsize=13)
plt.tight_layout()
plt.savefig(f'{SAVE_DIR}/03_correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()

# Flag high correlations (|r| > 0.85) for nb07 feature selection
high_corr = (
    corr.abs()
    .where(np.tril(np.ones(corr.shape), k=-1).astype(bool))
    .stack()
    .sort_values(ascending=False)
)
high_corr = high_corr[high_corr > 0.85]
if len(high_corr):
    print("High correlations (|r| > 0.85) -- candidates for nb07 review:")
    print(high_corr.to_string())
else:
    print("No feature pairs with |r| > 0.85")

# %%  [markdown]
# ## 4. Feature vs Cover Type (Box Plots)

# %%
plot_df = X_train[quant_cols].copy()
plot_df['Cover_Type'] = y_names.values

fig, axes = plt.subplots(4, 4, figsize=(22, 18))
axes = axes.flatten()
palette = ['#2196F3','#4CAF50','#FF9800','#9C27B0','#F44336','#00BCD4','#795548']

for i, col in enumerate(quant_cols):
    sns.boxplot(
        data=plot_df, x='Cover_Type', y=col,
        palette=palette, ax=axes[i],
        flierprops=dict(marker='.', markersize=1, alpha=0.3)
    )
    axes[i].set_title(col, fontsize=8)
    axes[i].set_xlabel('')
    axes[i].tick_params(axis='x', rotation=40, labelsize=6)

for j in range(len(quant_cols), len(axes)):
    axes[j].set_visible(False)

plt.suptitle('Quantitative Features by Cover Type', fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig(f'{SAVE_DIR}/04_features_by_class.png', dpi=150, bbox_inches='tight')
plt.close()

# %%  [markdown]
# ## 5. Wilderness Area vs Cover Type

# %%
# Recover Wilderness_Area as integer from OHE (argmax trick)
wld_df = X_train[ohe_wld_cols].idxmax(axis=1).str.extract(r'(\d+)$').astype(int)
wld_df.columns = ['Wilderness_Area']
wld_df['Cover_Type'] = y_names.values

ct = pd.crosstab(wld_df['Wilderness_Area'], wld_df['Cover_Type'])
ct_pct = ct.div(ct.sum(axis=1), axis=0) * 100

fig, axes = plt.subplots(1, 2, figsize=(16, 5))
ct.plot(kind='bar', ax=axes[0], colormap='tab10', edgecolor='none')
axes[0].set_title('Wilderness Area vs Cover Type (counts)')
axes[0].set_xlabel('Wilderness Area')
axes[0].tick_params(axis='x', rotation=0)
axes[0].legend(fontsize=7, title='Cover Type')

ct_pct.plot(kind='bar', stacked=True, ax=axes[1], colormap='tab10', edgecolor='none')
axes[1].set_title('Wilderness Area vs Cover Type (% within area)')
axes[1].set_xlabel('Wilderness Area')
axes[1].tick_params(axis='x', rotation=0)
axes[1].legend(fontsize=7, title='Cover Type')

plt.tight_layout()
plt.savefig(f'{SAVE_DIR}/05_wilderness_by_class.png', dpi=150, bbox_inches='tight')
plt.close()

# %%  [markdown]
# ## 6. Soil Type Sparsity

# %%
soil_counts = X_train[ohe_soil_cols].sum().sort_values(ascending=False)
sparse_threshold = len(X_train) * 0.01  # 1% of train rows
sparse_cols = soil_counts[soil_counts < sparse_threshold]

fig, ax = plt.subplots(figsize=(14, 5))
colors = ['#F44336' if v < sparse_threshold else '#2196F3' for v in soil_counts.values]
ax.bar(range(len(soil_counts)), soil_counts.values, color=colors, edgecolor='none')
ax.axhline(sparse_threshold, color='red', linestyle='--', linewidth=1.2,
           label=f'1% threshold ({int(sparse_threshold):,} rows)')
ax.set_title('Soil_Type OHE Column Frequency (Train Set)')
ax.set_xlabel('Soil Type column (sorted by frequency)')
ax.set_ylabel('Sample count')
ax.legend()
plt.tight_layout()
plt.savefig(f'{SAVE_DIR}/06_soil_sparsity.png', dpi=150, bbox_inches='tight')
plt.close()

print(f"Soil cols with < 1% rows ({int(sparse_threshold):,} samples): {len(sparse_cols)} of {len(ohe_soil_cols)}")
print("-> Strong nb07 drop candidates (importance-based selection will remove most of these)")
print(f"\nExtremely sparse (< 100 rows):")
print(sparse_cols[sparse_cols < 100].astype(int).to_string())

# %%  [markdown]
# ## 7. Engineered Feature Quality
#
# Do the 7 new features add class separation beyond the raw inputs?

# %%
from src.feature_engineering import ENGINEERED_COLS

eng_cols_present = [c for c in ENGINEERED_COLS if c in X_train.columns]
plot_df2 = X_train[eng_cols_present].copy()
plot_df2['Cover_Type'] = y_names.values

n = len(eng_cols_present)
ncols = 4
nrows = (n + ncols - 1) // ncols
fig, axes = plt.subplots(nrows, ncols, figsize=(22, 5 * nrows))
axes = axes.flatten()

for i, col in enumerate(eng_cols_present):
    sns.violinplot(
        data=plot_df2, x='Cover_Type', y=col,
        palette=palette, ax=axes[i], inner='quartile', linewidth=0.8
    )
    axes[i].set_title(col, fontsize=9)
    axes[i].set_xlabel('')
    axes[i].tick_params(axis='x', rotation=40, labelsize=6)

for j in range(n, len(axes)):
    axes[j].set_visible(False)

plt.suptitle('Engineered Features by Cover Type (violin)', fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig(f'{SAVE_DIR}/07_engineered_features.png', dpi=150, bbox_inches='tight')
plt.close()

# %%  [markdown]
# ## 7. EDA Summary

# %%
counts = y_train.value_counts().sort_index()
threshold = config['smote_threshold']

print("=== EDA Summary ===")
print(f"\nFeature matrix: {X_train.shape}")
print(f"  Quant features:     {len(quant_cols)}")
print(f"  Wilderness OHE:     {len(ohe_wld_cols)}")
print(f"  Soil_Type OHE:      {len(ohe_soil_cols)}")
print(f"  Total:              {X_train.shape[1]}")

print(f"\nClass counts and SMOTE decision (threshold = {threshold:,}):")
for cls, cnt in counts.items():
    decision = 'SMOTE' if cnt < threshold else 'no SMOTE'
    print(f"  {cls} {class_map[cls]:<22} {cnt:>7,}  -> {decision}")

smote_targets = [cls for cls, cnt in counts.items() if cnt < threshold]
print(f"\nSMOTE targets: classes {smote_targets}")
print(f"  -> {[class_map[c] for c in smote_targets]}")
print(f"  Class 5 Aspen ({counts[5]:,}) is ABOVE threshold -- no SMOTE")

print(f"\nMulticollinearity decision:")
print(f"  Hillshade_9am, Hillshade_Noon, Hillshade_3pm, Hillshade_mean are correlated.")
print(f"  Decision: KEEP all 4 -- tree models (RF/XGB/ET) are unaffected by multicollinearity.")
print(f"  Deferred to nb07 importance-based selection to decide which (if any) to drop.")

print(f"\nSoil type sparsity: 30 of {len(ohe_soil_cols)} cols have < 1% rows")
print(f"  Soil_Type_25 (1 row), Soil_Type_28 (6), Soil_Type_36 (9) are extremely sparse")
print(f"  -> nb07 importance threshold will drop most of these automatically")

print(f"\nFigures saved to: {SAVE_DIR}")
saved = [f for f in os.listdir(SAVE_DIR) if f.endswith('.png')]
for f in sorted(saved):
    print(f"  {f}")

# %%  [markdown]
# ## Summary
#
# | Finding | Detail |
# |---|---|
# | Class imbalance | 82,457x vs 1,728 ratio (47.7x); classes 3,4,6,7 = 1,728 rows each |
# | SMOTE targets | Classes 3,4,6,7 (< 2,000 threshold); class 5 Aspen (2,455) â€” NO SMOTE |
# | Multicollinearity | Hillshade_9am/Noon/3pm/mean correlated â€” KEEP all, defer to nb07 importance |
# | Soil type sparsity | 30/39 cols < 1% rows; 7 cols < 100 rows â€” nb07 will drop via importance threshold |
# | Top discriminators | Elevation, Hillshade features, Hydro distances (visible in box plots) |
# | Wilderness Area | Strong class predictor â€” certain cover types cluster in specific areas |
# | Engineered features | Elevation_Slope_interaction separates Krummholz; Hydro_Distance_Combined adds signal |
#
# **Next step -> 06 Imbalance Handling (SMOTE on classes 3, 4, 6, 7 only)**
