# EcoType — Forest Cover Type Classifier

## Model Description

- **Architecture:** XGBoost (champion; Optuna TPE-tuned, 20 trials)
- **Task:** 7-class multiclass classification of forest cover type from cartographic data
- **Primary metric:** Macro F1 (equal weight across all 7 classes regardless of frequency)
- **Champion performance:** Macro F1 = 0.9298 | Accuracy = 0.9658 | Weighted F1 = 0.9656
- **Test set:** 29,178 samples (stratified 20% hold-out from 145,891 total)

## Intended Use

- Forest management planning in the Rocky Mountain region (Colorado, USA)
- Land cover classification from elevation, slope, aspect, and distance measurements
- Educational demonstrations of end-to-end ML classification pipelines

## Out-of-Scope Use

- Real-time wildfire risk assessment
- Predictions outside the Colorado Rocky Mountain geography
- Financial, legal, or safety-critical decision-making
- Any application requiring predictions with confidence > 35% guarantees (model returns `low_confidence` flag below this threshold)

## Training Data

- **Source:** UCI ML Repository — Covertype Dataset (Roosevelt National Forest, Colorado, 1998)
- **Size:** 145,891 samples, 13 raw features → 59 engineered → ~50 selected by permutation importance
- **Target:** Cover_Type (1–7), 7 forest cover classes
- **Class imbalance:** Classes 1 and 2 account for ~85% of samples. SMOTE applied to classes 3, 4, 6, 7 to bring minority classes above 2,000 samples before training.
- **Feature engineering:** Aspect circular decomposition (sin/cos), Euclidean hydrology distance, elevation–slope interaction, hillshade mean, road/fire accessibility ratio

## Model Selection

Six models were trained and evaluated; Random Forest was selected as champion by highest macro F1 on the held-out test set:

| Model | Macro F1 | Accuracy | Weighted F1 | Role |
|-------|----------|----------|-------------|------|
| **XGBoost** | **0.9298** | **0.9658** | **0.9656** | **Champion** |
| Random Forest | 0.9146 | 0.9576 | — | Core ensemble |
| Extra Trees | 0.9070 | 0.9465 | — | Bonus ensemble |
| Decision Tree | 0.8665 | 0.9323 | 0.9331 | Interpretable |
| KNN | 0.8162 | 0.9048 | 0.9033 | Distance-based benchmark |
| Logistic Regression | 0.6391 | 0.6705 | 0.7039 | Baseline |

## Performance by Class (test set, XGBoost champion)

| Class | Name | Precision | Recall | F1-Score | Support |
|-------|------|-----------|--------|----------|---------|
| 1 | Spruce/Fir | 0.9524 | 0.9200 | 0.9359 | 6,222 |
| 2 | Lodgepole Pine | 0.9759 | 0.9840 | 0.9799 | 20,614 |
| 3 | Ponderosa Pine | 0.9007 | 0.8403 | 0.8695 | 432 |
| 4 | Cottonwood/Willow | 0.9332 | 0.9699 | 0.9512 | 432 |
| 5 | Aspen | 0.9250 | 0.9235 | 0.9242 | 614 |
| 6 | Douglas-fir | 0.8602 | 0.9259 | 0.8919 | 432 |
| 7 | Krummholz | 0.9338 | 0.9792 | 0.9559 | 432 |
| — | **macro avg** | **0.9259** | **0.9347** | **0.9298** | 29,178 |
| — | weighted avg | 0.9657 | 0.9658 | 0.9656 | 29,178 |

## Fairness Analysis (per Wilderness Area, XGBoost champion)

| Wilderness Area | Name | Macro F1 | n (test) | Note |
|----------------|------|----------|----------|------|
| 1 | Rawah | 0.5355 | 26,861 | Largest area; class imbalance within area drives lower macro F1 |
| 2 | Neota | 0.3220 | 92 | Very small sample — unreliable estimate |
| 3 | Comanche Peak | 0.7354 | 1,296 | Moderate performance |
| 4 | Cache la Poudre | 0.3870 | 929 | Low — dominated by classes with few test samples |

**Warning:** Per-wilderness macro F1 is substantially lower than overall (0.93) because minority classes (3, 4, 6, 7) are concentrated in specific areas where sample sizes are small. Do not use this model for standalone predictions in Areas 2 and 4.

## Known Limitations

- Trained on 1998 survey data — may not reflect current forest conditions after 25+ years of climate change and land use shifts
- Geographic scope: validated only for Roosevelt National Forest, Colorado. Performance outside this region is unknown.
- SMOTE-generated synthetic samples for rare classes (3, 4, 6, 7) may introduce ecological artifacts not present in real data
- Confidence threshold of 0.35 was set empirically, not calibrated on a validation set. Recalibrate before use in high-stakes decisions.
- n_jobs=-1 parallel training introduces ±0.001 macro F1 variance between runs (thread scheduling non-determinism)

## Inference Details

- **API:** POST `/predict` — 12 field input, returns cover_type_id, cover_type_name, confidence, probabilities (7 classes), low_confidence flag
- **Latency:** CPU-bound async endpoint; p50 latency ~30–80ms on Render free tier (cold start: 30–60s)
- **Rate limit:** 60 requests/minute per IP
- **Serialization:** joblib PKL stored on HuggingFace Hub, loaded at API startup

## MLflow Registry

- **Experiment:** `ecotype_forest_cover`
- **Registry name:** `ecotype_forest_cover`
- **Champion alias:** `models:/ecotype_forest_cover@champion`

## Contact

priya2359 — see [GitHub repository](https://github.com/priya2359) issue tracker  
Live API: https://ecotype-api.onrender.com  
Live UI: https://ecotype-forest-cover-type-prediction-using-machine-learning-tk.streamlit.app
