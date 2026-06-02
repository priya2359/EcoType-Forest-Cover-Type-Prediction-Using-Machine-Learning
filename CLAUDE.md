# EcoType — Claude Master Context

## Project
- Name: EcoType — Forest Cover Type Prediction
- Type: Multi-class Classification (7 classes)
- Dataset: 145,891 rows × 13 columns (GUVI subset)
- Target: Cover_Type (1-7)
- Python: 3.10.14 (venv locally; Render uses python-3.10.14 via runtime.txt)

## Progress Checklist
- [x] 01 Data Collection
- [x] 02 Data Understanding
- [x] 03 Data Cleaning
- [x] 04 Feature Engineering
- [x] 05 EDA
- [x] 06 Imbalance Handling
- [x] 07 Feature Selection
- [x] 08 Model Building + Hyperparameter Tuning (combined, FAST_MODE=False, Optuna 3-fold)
- [x] 09 Final Evaluation
- [x] FastAPI (api/) — live at https://ecotype-api.onrender.com
- [x] Streamlit (app/) — live at https://ecotype-forest-cover-type-prediction-using-machine-learning-tk.streamlit.app
- [x] Tests — 16/16 passing
- [x] CI/CD — GitHub Actions green (cache@v4, api-requirements.txt, smoke test)
- [x] Deploy: Render (API) + Streamlit Cloud (UI)
- [x] Post-audit production hardening (rate limiting, structured logging, confidence threshold)

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
- PKLs stored on HuggingFace Hub: priya2359/ecotype-forest-cover (never in git)
- Single master branch — no deploy branch needed
- Streamlit Cloud reads streamlit-requirements.txt (set in Advanced settings)
- Render env vars required: MODEL_SOURCE=huggingface, HF_REPO_ID, HF_TOKEN, PYTHONPATH=.
- Total features after OHE: 59 (16 quant + 4 wilderness + 39 soil)
- SMOTE targets: classes 3,4,6,7 only (1,728 → 2,455); class 5 Aspen (2,455) — no SMOTE
- Optuna: 3-fold StratifiedKFold always (hardcoded, not FAST_MODE dependent)
- Rate limiting: 60/minute via slowapi (api/limiter.py — single instance)
- API confidence threshold: 0.35 (env var CONFIDENCE_THRESHOLD)
- st.image() uses use_column_width=True (Streamlit version compatibility)

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
| EDA figures | `reports/figures/eda/` (01–09) |
| Model figures | `reports/figures/model/` (confusion matrices, calibration curves) |
| Cleaning figures | `reports/figures/cleaning/` |

## Live URLs
| Service | URL |
|---|---|
| FastAPI | https://ecotype-api.onrender.com |
| API Docs | https://ecotype-api.onrender.com/docs |
| Streamlit UI | https://ecotype-forest-cover-type-prediction-using-machine-learning-tk.streamlit.app |
| GitHub | https://github.com/priya2359/EcoType-Forest-Cover-Type-Prediction-Using-Machine-Learning |

## How to Resume Work
1. Read CLAUDE.md (this file) — check Progress Checklist
2. Activate venv: `.\venv\Scripts\Activate.ps1`
3. Run notebooks from PROJECT ROOT — never from inside notebooks/:
   `python notebooks/06_model_building.py`
4. API runs locally: `uvicorn api.main:app --reload`
5. UI runs locally: `streamlit run app/streamlit_app.py`
