# EcoType — Forest Cover Type Prediction

> **145,891 samples · 59 engineered features · 6 ML models · XGBoost champion (macro F1 = 0.9298) · FastAPI + Streamlit deployed**

![CI](https://github.com/priya2359/EcoType-Forest-Cover-Type-Prediction-Using-Machine-Learning/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=flat&logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-2.1.3-orange?style=flat)
![scikit--learn](https://img.shields.io/badge/scikit--learn-1.5.2-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?style=flat&logo=fastapi&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.40-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-2.14-0194E2?style=flat&logo=mlflow&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?style=flat&logo=docker&logoColor=white)

Predict which of 7 forest cover types occupies a 30×30m land patch using only cartographic measurements — no satellite imagery required. Built as a production ML system covering the complete lifecycle: raw data → feature engineering → Optuna-tuned models → async REST API → interactive Streamlit dashboard.

**Engineering highlights:** async inference with thread-pool offloading · single-pass `predict_proba` inference · structured JSON logging with request-ID tracing · SMOTE applied to training data only · `feature_columns.txt` artifact for exact training-serving alignment · HuggingFace Hub model storage · 25 automated tests · GitHub Actions CI

**Live demo:** [Streamlit App](https://ecotype-forest-cover-type-prediction-using-machine-learning-tk.streamlit.app) · [FastAPI Docs](https://ecotype-api.onrender.com/docs)

---

## What This Project Does

| Layer | What's Built |
|-------|-------------|
| **Data Pipeline** | 7-notebook workflow from raw CSV → schema validation → stratified split → feature selection |
| **Feature Engineering** | 7 domain features from cartographic inputs: circular aspect decomposition, Euclidean hydrology distance, elevation–slope interaction, hillshade mean, road/fire accessibility ratio |
| **Imbalance Handling** | Targeted SMOTE on classes 3, 4, 6, 7 (train only) — brings minority classes above 2,000 samples without contaminating the test set |
| **Feature Selection** | RandomForest permutation importance with 0.0005 threshold — 59 features reduced to 44; exact column order saved to `artifacts/feature_columns.txt` |
| **ML Models** | 6 models: Logistic Regression, Decision Tree, KNN, Random Forest, Extra Trees, XGBoost — Optuna TPE tuning for RF, ET, XGB |
| **Model Registry** | MLflow versioned registry with `champion` alias — XGBoost promoted after final evaluation |
| **API** | FastAPI with async prediction endpoint, rate limiting (60 req/min), structured JSON logging, request-ID tracing, `FeatureEngineeringError` → HTTP 422 |
| **Dashboard** | Streamlit app with 3 pages: Prediction, EDA Dashboard, Model Insights |
| **Model Storage** | PKLs stored on HuggingFace Hub — downloaded once at API startup, not committed to git |
| **Tests** | 25 automated tests — API endpoints, preprocessing, feature engineering edge cases · GitHub Actions CI |

---

## Architecture

```
Raw CSV (145,891 rows × 13 features)
         │
         ▼
  7-Notebook Pipeline
  ┌──────────────────────────────────────────────────┐
  │ 01 clean → 02 engineer → 03 eda → 04 smote       │
  │ 05 select → 06 train+tune → 07 evaluate          │
  └──────────────────────────────────────────────────┘
         │
         ▼
  Artifacts (pkl + txt)
  ├── models/best_model/ecotype_best_model.pkl  ──► HuggingFace Hub
  ├── models/encoders/  (wilderness_ohe, soil_ohe)
  ├── models/scalers/   (standard_scaler)
  └── artifacts/        (feature_columns.txt, quant_feature_columns.txt)
         │
         ▼  (loaded once at startup)
  FastAPI (Render)
  └── POST /predict
       raw input
         │
         ▼  async run_in_executor (thread pool)
       engineer_features()
         → OHE (Wilderness_Area, Soil_Type)
         → StandardScaler (quant cols only)
         → reindex to feature_columns.txt order
         → XGBoost.predict_proba()
         │
         ▼
      JSON response (class, confidence, probabilities[7], low_confidence flag)
         │
         ▼
  Streamlit (Streamlit Cloud)
  ├── Page 1 — Prediction   (12-field form, live API call, bar chart)
  ├── Page 2 — EDA Dashboard (figures from reports/figures/eda/)
  └── Page 3 — Model Insights (comparison table, confusion matrices, Optuna trials)
```

---

## ML Models

### 1. XGBoost *(Champion — macro F1 = 0.9298)*
- **Tuning:** Optuna TPE, 50 trials, 5-fold stratified CV, MedianPruner
- **Search space:** `n_estimators` [200/300/500], `max_depth` [4–8], `learning_rate` [0.05–0.2], `subsample` [0.7–1.0], `colsample_bytree` [0.7–1.0]
- **Label handling:** `XGBWrapper` shifts Cover_Type 1–7 → 0–6 on `fit()`, restores on `predict()` via `classes_` attribute — fully sklearn-compatible
- **Inference:** single `predict_proba()` call; `class_id = classes_[argmax(proba)]` — no redundant `predict()` call

### 2. Random Forest *(macro F1 = 0.9146)*
- **Tuning:** Optuna TPE, 50 trials — `n_estimators`, `max_depth`, `min_samples_leaf`, `max_features`
- `class_weight="balanced"`, `n_jobs=-1`

### 3. Extra Trees *(macro F1 = 0.9070)*
- Same search space as Random Forest; higher variance, faster training
- `class_weight="balanced"`, `n_jobs=-1`

### 4. Decision Tree *(macro F1 = 0.8665)*
- No tuning — interpretable baseline; `max_depth=20`, `class_weight="balanced"`

### 5. KNN *(macro F1 = 0.8162)*
- `ball_tree` algorithm, `k=5` — evaluation only, not deployed to API

### 6. Logistic Regression *(macro F1 = 0.6391)*
- Linear baseline; `lbfgs` solver, `multinomial`, `class_weight="balanced"` — not tuned

---

## Results

| Model | Macro F1 | Accuracy | Weighted F1 | Role |
|-------|----------|----------|-------------|------|
| **XGBoost** | **0.9298** | **0.9658** | **0.9656** | **Champion** |
| Random Forest | 0.9146 | 0.9576 | 0.9577 | Core ensemble |
| Extra Trees | 0.9070 | 0.9465 | 0.9472 | Bonus ensemble |
| Decision Tree | 0.8665 | 0.9323 | 0.9331 | Interpretable |
| KNN | 0.8162 | 0.9048 | 0.9033 | Distance-based benchmark |
| Logistic Regression | 0.6391 | 0.6705 | 0.7039 | Baseline |

**Per-class F1 (XGBoost, test set):**

| Class | Name | F1 | Support |
|-------|------|----|---------|
| 2 | Lodgepole Pine | 0.9799 | 20,614 |
| 7 | Krummholz | 0.9559 | 432 |
| 4 | Cottonwood/Willow | 0.9512 | 432 |
| 1 | Spruce/Fir | 0.9359 | 6,222 |
| 5 | Aspen | 0.9242 | 614 |
| 6 | Douglas-fir | 0.8919 | 432 |
| 3 | Ponderosa Pine | 0.8695 | 432 |

---

## Tech Stack

```
Python 3.10          XGBoost 2.1.3        scikit-learn 1.5.2
imbalanced-learn     Optuna 4.8           MLflow 2.14
FastAPI 0.115        Pydantic 2.8         SlowAPI 0.1.9
Streamlit 1.40       Plotly 5.22          HuggingFace Hub
Docker Compose       pytest 8.3           httpx 0.27
PyYAML 6.0           joblib 1.4           uvicorn 0.30
```

---

## Project Structure

```
├── notebooks/
│   ├── initial_analysing.ipynb       # Dataset reference — column types, value ranges
│   ├── 01_data_cleaning.py           # Outlier inspection, schema validation, stratified split
│   ├── 02_feature_engineering.py     # 7 domain features; Aspect circular decomposition
│   ├── 03_eda.py                     # 9 EDA charts saved to reports/figures/eda/
│   ├── 04_imbalance_handling.py      # SMOTE on minority classes (train only)
│   ├── 05_feature_selection.py       # Permutation importance → feature_columns.txt
│   ├── 06_model_building.py          # Train 6 models; Optuna TPE for RF, ET, XGB
│   └── 07_final_evaluation.py        # Champion selection, calibration curves, MLflow registry
├── src/
│   ├── data_loader.py                # CSV load, schema validation, stratified split
│   ├── preprocessing.py              # OHE (fit/transform), StandardScaler, validate_schema
│   ├── feature_engineering.py        # engineer_features() + ENGINEERED_COLS constant
│   ├── feature_selector.py           # Importance-based selection, save/load feature artifacts
│   ├── imbalance_handler.py          # SMOTE wrapper with explicit sampling_strategy dict
│   ├── model_trainer.py              # train_model(), log_run_to_mlflow() with training time
│   ├── hyperparameter_tuner.py       # Optuna objectives for RF / ET / XGB; _get_cv_folds()
│   ├── model_evaluator.py            # evaluate(), confusion matrix, model comparison chart
│   ├── predictor.py                  # load_artifacts(), preprocess_input(), predict()
│   └── xgb_wrapper.py                # sklearn-compatible XGBoost with 1–7 label offset
├── api/
│   ├── main.py                       # FastAPI lifespan model loading, JSON logging, FeatureEngineeringError handler
│   ├── limiter.py                    # SlowAPI rate limiter singleton (60 req/min)
│   ├── middleware/logging_middleware.py  # Structured JSON logs, request-ID tracing
│   ├── routes/health.py              # GET /health
│   ├── routes/predict.py             # POST /predict (async, thread-pool offloaded)
│   └── schemas/input_schema.py       # Pydantic v2 — validated ranges for all 12 fields
├── app/
│   ├── streamlit_app.py              # Home page, sidebar API config
│   ├── pages/
│   │   ├── 1_Prediction.py           # 12-field form → API call → bar chart
│   │   ├── 2_EDA_Dashboard.py        # EDA figures from reports/figures/eda/
│   │   └── 3_Model_Insights.py       # Comparison table, confusion matrices, Optuna trials
│   └── utils/offline_predictor.py    # Local fallback with lru_cache; returns None on Streamlit Cloud
├── configs/
│   ├── feature_config.yaml           # Feature lists, class_map, SMOTE threshold, FE formulas
│   ├── model_config.yaml             # Hyperparameters, Optuna config (trials, cv_folds)
│   └── app_config.yaml               # API host/port, Streamlit title
├── artifacts/
│   ├── feature_columns.txt           # Exact 44-column training order for inference alignment
│   └── quant_feature_columns.txt     # 16 quantitative column names for StandardScaler
├── models/                           # PKL artifacts — excluded from git, stored on HuggingFace Hub
│   ├── best_model/ecotype_best_model.pkl
│   ├── encoders/  (wilderness_ohe.pkl, soil_ohe.pkl)
│   └── scalers/   (standard_scaler.pkl)
├── reports/
│   ├── figures/eda/                  # 9 EDA charts (class distribution, correlations, SMOTE)
│   ├── figures/model/                # 7 confusion matrices + calibration curves + comparison bar
│   ├── figures/cleaning/             # Before/after outlier histograms
│   └── optuna/                       # Trial CSVs for RF, ET, XGB
├── tests/
│   ├── conftest.py                   # Mock artifacts fixture for API tests (runs in CI without PKLs)
│   ├── test_api.py                   # 6 API endpoint tests
│   ├── test_preprocessing.py         # 6 OHE + scaler + schema tests
│   └── test_feature_engineering.py   # 9 idempotency, determinism, and edge-case tests
├── docker/
│   ├── Dockerfile.api                # Python 3.10-slim, non-root appuser
│   └── Dockerfile.streamlit          # Python 3.10-slim, non-root appuser
├── .github/workflows/ci.yml          # Python 3.10, pip cache, smoke tests + pytest on every push
├── docker-compose.yml                # API + Streamlit services with health checks
├── render.yaml                       # Render deployment config (HuggingFace model source)
├── MODEL_CARD.md                     # Model description, per-class F1, fairness analysis, limitations
└── requirements.txt / api-requirements.txt / streamlit-requirements.txt
```

---

## Quick Start

### Option A — Use the live deployment (no setup needed)

```bash
# Predict directly from the API:
curl -X POST https://ecotype-api.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Elevation": 2596, "Aspect": 51, "Slope": 3,
    "Horizontal_Distance_To_Hydrology": 258, "Vertical_Distance_To_Hydrology": 0,
    "Horizontal_Distance_To_Roadways": 510, "Hillshade_9am": 221,
    "Hillshade_Noon": 232, "Hillshade_3pm": 148,
    "Horizontal_Distance_To_Fire_Points": 6279,
    "Wilderness_Area": 1, "Soil_Type": 29
  }'
```

> **Note:** First request on Render free tier may take 30–60s (cold start). The API downloads the model from HuggingFace Hub on first boot.

---

### Option B — Local development

**Prerequisites:** Python 3.10, git

```bash
git clone https://github.com/priya2359/EcoType-Forest-Cover-Type-Prediction-Using-Machine-Learning.git
cd EcoType-Forest-Cover-Type-Prediction-Using-Machine-Learning

python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Mac/Linux

pip install -r requirements.txt
cp .env.example .env           # edit MODEL_SOURCE=local if you have PKLs
```

**Run the API:**
```bash
uvicorn api.main:app --reload
# API docs → http://localhost:8000/docs
```

**Run the Streamlit app** (in a second terminal):
```bash
streamlit run app/streamlit_app.py
# Dashboard → http://localhost:8501
```

**Run tests:**
```bash
pytest tests/ -v
# 25 tests pass — no PKLs required (API + preprocessing + feature engineering tests use mocks)
```

---

### Option C — Docker (API + UI together)

```bash
docker-compose up --build
# API → :8000   Streamlit → :8501
```

---

### Re-run the full ML pipeline

Requires the raw dataset at `data/raw/cover_type (1).csv`:

```bash
python notebooks/01_data_cleaning.py
python notebooks/02_feature_engineering.py
python notebooks/03_eda.py
python notebooks/04_imbalance_handling.py
python notebooks/05_feature_selection.py
python notebooks/06_model_building.py   # FAST_MODE=False for full Optuna (~2 hrs)
python notebooks/07_final_evaluation.py
```

Track experiments: `mlflow ui --port 5000`

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Service health + model load status |
| POST | `/predict` | Predict forest cover type from 12 cartographic fields |
| GET | `/docs` | Interactive Swagger UI |

**POST `/predict` — request body (all fields required):**

| Field | Type | Valid range |
|-------|------|------------|
| `Elevation` | int | 1859 – 3858 m |
| `Aspect` | int | 0 – 360° |
| `Slope` | int | 0 – 66° |
| `Horizontal_Distance_To_Hydrology` | int | ≥ 0 m |
| `Vertical_Distance_To_Hydrology` | int | any (can be negative) |
| `Horizontal_Distance_To_Roadways` | int | ≥ 0 m |
| `Hillshade_9am` | int | 0 – 255 |
| `Hillshade_Noon` | int | 0 – 255 |
| `Hillshade_3pm` | int | 0 – 255 |
| `Horizontal_Distance_To_Fire_Points` | int | ≥ 0 m |
| `Wilderness_Area` | int | 1 – 4 |
| `Soil_Type` | int | 1 – 40 |

**Response:** `cover_type_id`, `cover_type_name`, `confidence`, `probabilities` (all 7 classes), `low_confidence` flag, `model_version`

---

## Tests

```bash
pytest tests/ -v                    # 25 tests
pytest tests/test_api.py -v         # 6 API endpoint tests (mock artifacts — runs in CI without PKLs)
pytest tests/test_preprocessing.py  # 6 OHE + scaler + schema tests
pytest tests/test_feature_engineering.py  # 9 idempotency, determinism, edge-case tests
pytest tests/test_model.py          # auto-skips in CI when PKLs not present
```

CI runs on every push to `master` and `main` via GitHub Actions.

---

## Production Readiness

| Category | What's Implemented |
|---|---|
| **Performance** | Single-pass `predict_proba` inference — `class_id` derived from `classes_[argmax(proba)]`, no redundant `predict()` call |
| **Concurrency** | Async endpoint with `run_in_executor` (thread pool) — event loop stays free during CPU-bound sklearn work |
| **Robustness** | `FeatureEngineeringError` custom exception → HTTP 422 · NaN guard after feature engineering · 503 when model not loaded |
| **Observability** | Structured JSON logging on every request · request-ID threaded from middleware into prediction log · `LOG_LEVEL` env var |
| **Security** | CORS with explicit allowed origins · rate limiting (60/min) · no secrets in VCS · non-root Docker user |
| **Reliability** | `feature_columns.txt` artifact prevents training-serving feature mismatch · `handle_unknown=ignore` in OHE for unseen categories |
| **Reproducibility** | `random_state=42` throughout · pinned dependency versions in 3 split requirements files |
| **Portability** | `XGBWrapper.classes_` attribute makes XGBoost drop-in compatible with all sklearn model code |
| **Tests** | 25 automated tests · mock fixtures let API tests run in CI without model PKLs |

---

## Key Design Decisions

- **SMOTE on training data only:** Fit SMOTE after the train/test split, never before. Oversampling before splitting causes data leakage — synthetic minority samples end up in the test set, inflating metrics.
- **Circular aspect decomposition:** Raw `Aspect` is an integer 0–360°. A value of 1° and 359° are geographically 2° apart but numerically 358 apart — breaking any distance-based or gradient calculation. `sin(deg2rad(Aspect))` + `cos(deg2rad(Aspect))` preserves true compass bearing.
- **`feature_columns.txt` for inference alignment:** After feature selection and OHE expansion, the model expects 44 columns in a specific order. Saving this order to a text file and calling `df.reindex(columns=feature_cols)` at inference time prevents silent column-order mismatches across environments.
- **`XGBWrapper` label offset:** XGBoost requires 0-indexed labels (0–6) but Cover_Type is 1–7. The wrapper subtracts 1 on `fit()` and stores original labels in `classes_`. This makes the wrapper a drop-in sklearn estimator — all downstream code (evaluator, API, tests) uses standard sklearn API without any offset awareness.
- **Single-pass inference:** `predict()` calls `predict_proba(X)` once and derives `class_id = model.classes_[argmax(proba)]` — never calls `predict(X)` separately. For a 200-estimator Random Forest, a redundant `predict()` call doubles inference time per request.
- **Optuna MedianPruner:** Stops unpromising trials after the first 3 folds if their median is below the best seen so far — reduces total tuning time by ~30% without affecting final result quality.
- **Async endpoint with `run_in_executor`:** FastAPI's default `def` handlers run in a thread pool automatically, but using `async def` with explicit `run_in_executor` gives explicit control over concurrency and avoids the thread-pool exhaustion that occurs under high load with synchronous CPU-bound handlers.
- **Split requirements files:** `requirements.txt` (full dev), `api-requirements.txt` (Render — no MLflow/Streamlit), `streamlit-requirements.txt` (Streamlit Cloud — no FastAPI). Keeps production images lean and avoids dependency conflicts between stacks.

---

## Dataset

| Source | Rows | Features | Period |
|--------|------|----------|--------|
| [UCI ML Repository — Covertype](https://archive.ics.uci.edu/dataset/31/covertype) | 145,891 | 13 raw → 59 engineered → 44 selected | 1998 survey |

**Forest cover classes:** Spruce/Fir · Lodgepole Pine · Ponderosa Pine · Cottonwood/Willow · Aspen · Douglas-fir · Krummholz

**Blackard, J. & Dean, D. (1999).** *Covertype.* UCI Machine Learning Repository. https://doi.org/10.24432/C50K5N

---

## Author

**Priya Neha** · [GitHub](https://github.com/priya2359)
