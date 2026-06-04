# EcoType — Forest Cover Type Prediction

Predict which of 7 forest cover types is present in a 30×30m land patch using cartographic measurements — no satellite imagery required. Built as a production ML system with a REST API backend and an interactive Streamlit frontend.

**Live demo:** [Streamlit App](https://ecotype-forest-cover-type-prediction-using-machine-learning-tk.streamlit.app) · [FastAPI](https://ecotype-api.onrender.com/docs)

---

## Results

| Model | Macro F1 | Accuracy | Role |
|-------|----------|----------|------|
| **XGBoost** *(champion)* | **0.9298** | **0.9658** | Optuna-tuned |
| Random Forest | 0.9146 | 0.9576 | Optuna-tuned |
| Extra Trees | 0.9070 | 0.9465 | Optuna-tuned |
| Decision Tree | 0.8665 | 0.9323 | Interpretable baseline |
| KNN | 0.8162 | 0.9048 | Distance-based benchmark |
| Logistic Regression | 0.6391 | 0.6705 | Linear baseline |

XGBoost outperformed Random Forest after Optuna tuning — improving from a 0.9043 baseline to **0.9298 macro F1**, the highest across all six models.

---

## Dataset

**Source:** [UCI ML Repository — Covertype](https://archive.ics.uci.edu/dataset/31/covertype)  
**Size:** 145,891 samples × 13 features  
**Target:** `Cover_Type` — 7 forest cover classes  
**Location:** Roosevelt National Forest, Colorado, USA (1998 cartographic survey)

| Class | Name | Train samples |
|-------|------|--------------|
| 1 | Spruce/Fir | 63,168 |
| 2 | Lodgepole Pine | 48,373 |
| 3 | Ponderosa Pine | 6,101 |
| 4 | Cottonwood/Willow | 2,160 |
| 5 | Aspen | 9,493 |
| 6 | Douglas-fir | 17,367 |
| 7 | Krummholz | 20,510 |

Classes 3, 4, and 6 are severely underrepresented — handled with targeted SMOTE oversampling on training data only.

---

## Approach

### Pipeline (7 notebooks)

```
01_data_cleaning      → outlier inspection, schema validation, stratified train/test split
02_feature_engineering → 7 new features from domain knowledge (see below)
03_eda                → univariate/bivariate analysis, class and wilderness distributions
04_imbalance_handling → targeted SMOTE on classes 3, 4, 6, 7 (train only, never test)
05_feature_selection  → permutation importance threshold → 44 features kept from 59
06_model_building     → train 6 models; Optuna TPE tuning for RF, ET, XGBoost
07_final_evaluation   → champion selection, classification report, calibration curves
```

### Feature Engineering

Raw `Aspect` (0–360°) is circular — 1° and 359° are 2° apart but differ by 358 as integers. Decomposed into sin/cos components to preserve true compass-bearing distance.

| Feature | Formula | Rationale |
|---------|---------|-----------|
| `Aspect_sin` | sin(deg2rad(Aspect)) | Circular decomposition |
| `Aspect_cos` | cos(deg2rad(Aspect)) | Circular decomposition |
| `Hydro_Distance_Combined` | √(H_hydro² + V_hydro²) | True Euclidean distance to water |
| `Hydro_Elev_interaction` | Elevation − V_hydro | Absolute elevation of nearest water |
| `Hillshade_mean` | mean(9am, noon, 3pm) | Average daily solar exposure |
| `Elevation_Slope_interaction` | Elevation × Slope | High elevation + steep slope → Krummholz |
| `Distance_Road_Fire_Ratio` | H_road / (H_fire + 1) | Relative accessibility vs fire proximity |

### Imbalance Handling

SMOTE was applied **only to training data**, after the train/test split, to bring minority classes to a minimum of 2,000 samples. Test set class proportions were never altered.

### Hyperparameter Tuning

Optuna TPE (Tree-structured Parzen Estimator) with 5-fold stratified CV and MedianPruner. 50 trials each for Random Forest, Extra Trees, and XGBoost.

---

## Per-Class Performance (XGBoost champion)

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| Spruce/Fir | 0.9524 | 0.9200 | 0.9359 | 6,222 |
| Lodgepole Pine | 0.9759 | 0.9840 | **0.9799** | 20,614 |
| Ponderosa Pine | 0.9007 | 0.8403 | 0.8695 | 432 |
| Cottonwood/Willow | 0.9332 | 0.9699 | 0.9512 | 432 |
| Aspen | 0.9250 | 0.9235 | 0.9242 | 614 |
| Douglas-fir | 0.8602 | 0.9259 | 0.8919 | 432 |
| Krummholz | 0.9338 | 0.9792 | 0.9559 | 432 |

Ponderosa Pine (class 3) and Douglas-fir (class 6) are the hardest to classify — both are rare in the dataset and ecologically similar to neighbouring cover types.

---

## Key Insights

- **Elevation dominates** — the single most predictive feature. Krummholz only exists above ~3,400m; Cottonwood/Willow clusters near water at lower elevations.
- **SMOTE is necessary** — without it, the model ignores minority classes entirely due to the 63:1 class ratio between Lodgepole Pine and Cottonwood/Willow.
- **XGBoost wins after tuning** — at default settings XGBoost (0.9043) trailed Random Forest (0.9149). After Optuna tuning, XGBoost surpassed all models at 0.9298.
- **Wilderness area matters** — per-area macro F1 ranges from 0.32 (Neota, only 92 test samples) to 0.74 (Comanche Peak). Model is least reliable in under-sampled wilderness areas.
- **Circular aspect encoding works** — raw Aspect caused discontinuities at 0°/360°. Sin/cos decomposition improved separation between north-facing and south-facing slopes.

---

## Architecture

```
┌─────────────────────┐        POST /predict        ┌──────────────────────┐
│   Streamlit UI      │  ─────────────────────────► │   FastAPI (Render)   │
│  (Streamlit Cloud)  │ ◄─────────────────────────  │   + XGBoost model    │
└─────────────────────┘      JSON response           └──────────────────────┘
                                                               │
                                                      loads pkl on startup
                                                               │
                                                     ┌──────────────────────┐
                                                     │   HuggingFace Hub    │
                                                     │  (model storage)     │
                                                     └──────────────────────┘
```

**Inference pipeline** (training order preserved exactly):
`raw input → feature engineering → OHE (Wilderness, Soil) → StandardScaler → feature alignment → XGBoost`

---

## Project Structure

```
├── notebooks/          # 7-step ML pipeline (# %% format, runnable as scripts)
├── src/                # Core pipeline modules (data_loader, preprocessor, trainer, etc.)
├── api/                # FastAPI backend with rate limiting, structured logging
├── app/                # Streamlit UI (3 pages: Prediction, EDA, Model Insights)
├── configs/            # YAML config (features, model hyperparams, class map)
├── models/             # PKL artifacts (stored on HuggingFace Hub, not in git)
├── artifacts/          # feature_columns.txt, quant_feature_columns.txt
├── reports/figures/    # EDA plots, confusion matrices, calibration curves
├── tests/              # pytest suite — API, preprocessing, feature engineering
├── docker/             # Dockerfiles for API and Streamlit containers
├── MODEL_CARD.md       # Model description, limitations, fairness analysis
└── render.yaml         # Render deployment config
```

---

## Run Locally

**Prerequisites:** Python 3.10, git

```bash
git clone https://github.com/priya2359/EcoType-Forest-Cover-Type-Prediction-Using-Machine-Learning.git
cd EcoType-Forest-Cover-Type-Prediction-Using-Machine-Learning

python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Mac/Linux

pip install -r requirements.txt
```

**Run the API:**
```bash
uvicorn api.main:app --reload
# API docs at http://localhost:8000/docs
```

**Run the Streamlit app:**
```bash
# In a second terminal, with venv active:
streamlit run app/streamlit_app.py
```

**Run the ML pipeline** (retrains all 6 models — takes ~2 hours):
```bash
python notebooks/01_data_cleaning.py
python notebooks/02_feature_engineering.py
python notebooks/03_eda.py
python notebooks/04_imbalance_handling.py
python notebooks/05_feature_selection.py
python notebooks/06_model_building.py   # set FAST_MODE=False for full Optuna
python notebooks/07_final_evaluation.py
```

**Run tests:**
```bash
pytest tests/ -v
```

---

## Technologies

| Category | Tools |
|----------|-------|
| ML | scikit-learn, XGBoost, imbalanced-learn |
| Tuning | Optuna (TPE sampler, MedianPruner) |
| Tracking | MLflow |
| API | FastAPI, Pydantic, SlowAPI |
| UI | Streamlit, Plotly |
| Model storage | HuggingFace Hub |
| Deployment | Render (API), Streamlit Cloud (UI) |
| Testing | pytest, httpx |
| Containers | Docker, docker-compose |

---

## Dataset Citation

Blackard, J. & Dean, D. (1999). *Covertype*. UCI Machine Learning Repository. [https://doi.org/10.24432/C50K5N](https://doi.org/10.24432/C50K5N)
