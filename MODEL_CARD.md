# EcoType — Forest Cover Type Classifier

## Model Description

- **Architecture:** Random Forest (champion; Optuna TPE-tuned, 50 trials)
- **Task:** 7-class multiclass classification of forest cover type from cartographic data
- **Primary metric:** Macro F1 (equal weight across all 7 classes regardless of frequency)
- **Baseline performance:** Macro F1 = 0.9149 | Accuracy = 0.9582 | Weighted F1 = 0.9577
- **Test set:** 29,179 samples (stratified 20% hold-out from 145,891 total)

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
| **Random Forest** | **0.9149** | **0.9582** | **0.9577** | Champion |
| Extra Trees | 0.9062 | 0.9481 | 0.9472 | Ensemble |
| XGBoost | 0.9043 | 0.9424 | 0.9414 | Primary candidate |
| Decision Tree | 0.8665 | 0.9323 | 0.9331 | Interpretable |
| KNN | 0.8162 | 0.9048 | 0.9033 | Distance-based benchmark |
| Logistic Regression | 0.6391 | 0.6705 | 0.7039 | Baseline |

## Performance by Class (test set)

<!-- Fill in from notebooks/07_final_evaluation.py classification report output -->
<!-- Run: python notebooks/07_final_evaluation.py and copy the per-class F1 values below -->

| Class | Name | Precision | Recall | F1-Score | Support |
|-------|------|-----------|--------|----------|---------|
| 1 | Spruce/Fir | TODO | TODO | TODO | TODO |
| 2 | Lodgepole Pine | TODO | TODO | TODO | TODO |
| 3 | Ponderosa Pine | TODO | TODO | TODO | TODO |
| 4 | Cottonwood/Willow | TODO | TODO | TODO | TODO |
| 5 | Aspen | TODO | TODO | TODO | TODO |
| 6 | Douglas-fir | TODO | TODO | TODO | TODO |
| 7 | Krummholz | TODO | TODO | TODO | TODO |

## Fairness Analysis (per Wilderness Area)

<!-- Fill in from notebooks/07_final_evaluation.py per-wilderness breakdown -->

| Wilderness Area | Name | Macro F1 | n (test) |
|----------------|------|----------|----------|
| 1 | Rawah | TODO | TODO |
| 2 | Neota | TODO | TODO |
| 3 | Comanche Peak | TODO | TODO |
| 4 | Cache la Poudre | TODO | TODO |

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
