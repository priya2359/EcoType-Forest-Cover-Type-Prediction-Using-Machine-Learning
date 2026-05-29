# EcoType — Claude Master Context

## Project
- Name: EcoType — Forest Cover Type Prediction
- Type: Multi-class Classification (7 classes)
- Dataset: 145,891 rows × 13 columns (GUVI subset)
- Target: Cover_Type (1-7)
- Python: 3.10.11 (venv)

## Progress Checklist
- [x] 01 Data Collection
- [x] 02 Data Understanding
- [x] 03 Data Cleaning
- [x] 04 Feature Engineering
- [x] 05 EDA
- [x] 06 Imbalance Handling
- [x] 07 Feature Selection
- [x] 06 Model Building + Hyperparameter Tuning (combined, FAST_MODE=False complete)
- [x] 07 Final Evaluation
- [x] FastAPI (api/) — verified live
- [x] Streamlit (app/) — verified live, st.image fix applied
- [x] Tests — 16/16 passing
- [ ] Docker — .dockerignore added, build pending
- [ ] CI/CD
- [ ] Deploy: Render (API) + Streamlit Cloud (UI)

## Golden Rules (Never Break These)
1. SMOTE only on training data — never before split
2. fit() / fit_transform() on train only — transform() everywhere else
3. Use get_feature_names_out() for OHE column names — never hardcode
4. from src.xgb_wrapper import XGBWrapper # noqa — before every joblib.load()
5. Macro F1 is primary metric — accuracy is secondary
6. FAST_MODE=False before running notebook 06 for final models
7. Save each model immediately after training — never batch without saves
8. Use try/except NameError for __file__ in notebooks — not __file__ directly
9. Log ALL errors to reports/ERROR_LOG.md after every section

## Key Decisions
- XGBWrapper handles Cover_Type 1-7 label offset (XGB expects 0-6)
- feature_columns.txt = exact training column order for inference alignment
- quant_feature_columns.txt = quant col names for scaler at inference
- cover_type_map single source of truth = feature_config.yaml class_map
- PKLs stored on HuggingFace Hub (never in git — >100MB limit)
- Single main branch; no deploy branch needed
- Streamlit Cloud: set "Requirements file" to streamlit-requirements.txt in dashboard
- Total features after nb04 OHE: 59 (16 quant + 4 wilderness + 39 soil)
- SMOTE targets (nb06): classes 3,4,6,7 only (1,728 rows each, below 2,000 threshold); class 5 Aspen (2,455) — no SMOTE
- SMOTE output: data/processed/X_train_smote.csv · data/processed/y_train_smote.csv (nb07 loads these exact names)
- 30/39 soil OHE cols have < 1% rows — nb07 importance threshold will drop most automatically
- Soil_Type OHE = 39 cols (one type absent from train split); API keeps Field(ge=1, le=40) — missing type returns zeros via handle_unknown='ignore'
- nb07 feature selection starts with 59 features; expect 50-57 after dropping low-importance soil cols; feature_columns.txt has exact final count

## Notebook Registry
| File | Status |
|---|---|
| initial_analysing.ipynb | Dataset reference — no number |
| 01_data_cleaning.py | Complete |
| 02_feature_engineering.py | Complete |
| 03_eda.py | Complete |
| 04_imbalance_handling.py | Complete |
| 05_feature_selection.py | Complete |
| 06_model_building.py | Complete (FAST_MODE=False, Optuna 3-fold) |
| 07_final_evaluation.py | Complete |

## Artifact Locations
| Artifact | Path |
|---|---|
| MLflow runs + registry | `mlruns/` (project root — NEVER inside notebooks/) |
| OHE encoders | `models/encoders/wilderness_ohe.pkl` · `soil_ohe.pkl` |
| Scaler | `models/scalers/standard_scaler.pkl` |
| Trained models | `models/trained/{name}.pkl` |
| Best model | `models/best_model/ecotype_best_model.pkl` |
| Feature columns | `artifacts/feature_columns.txt` · `quant_feature_columns.txt` |
| Optuna trials | `reports/optuna/random_forest_trials.csv` · `extra_trees_trials.csv` · `xgboost_trials.csv` |

## How to Resume Work
1. Read CLAUDE.md (this file) — check Progress Checklist
2. Read reports/ERROR_LOG.md — review past errors
3. Activate venv: `.\venv\Scripts\Activate.ps1`
4. Run notebooks from PROJECT ROOT — never from inside notebooks/:
   `python notebooks/07_final_evaluation.py`
5. Continue from last unchecked step
