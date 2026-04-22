# Architecture Overview

## Project Structure

```
football-match-prediction/
├── README.md                  # Project overview & quick-start
├── requirements.txt           # Python dependencies
├── setup.py                   # Package installation
├── Makefile                   # Common commands (train, test, lint, docker)
│
├── configs/
│   └── config.yaml            # Central configuration (data, features, models, API)
│
├── data/
│   ├── raw/                   # Downloaded CSVs (not committed)
│   ├── processed/             # Cleaned & feature-engineered CSVs (not committed)
│   └── external/              # Third-party datasets (StatsBomb, etc.)
│
├── src/
│   ├── data/
│   │   ├── ingestion.py       # Download data from Football-Data.co.uk / StatsBomb
│   │   ├── preprocessing.py   # Clean, rename columns, encode target label
│   │   └── pipeline.py        # Orchestrate: ingest → preprocess → features
│   ├── features/
│   │   └── engineering.py     # Elo ratings, rolling form, time-decay weighting
│   ├── models/
│   │   ├── train.py           # Train LogReg / RF / XGBoost / LightGBM, log to MLflow
│   │   ├── evaluate.py        # Metrics (accuracy, log-loss, Brier), SHAP explanations
│   │   └── predict.py         # Load saved model and produce predictions
│   └── utils/
│       └── helpers.py         # Config loading, seed, logging, path helpers
│
├── api/
│   ├── main.py                # FastAPI app (lifespan, CORS, endpoints)
│   └── schemas.py             # Pydantic v2 request/response models
│
├── tests/
│   ├── test_data.py           # Unit tests for ingestion & preprocessing
│   ├── test_features.py       # Unit tests for Elo, form, rolling stats
│   ├── test_models.py         # Unit tests for metrics, prepare_data, end-to-end
│   └── test_api.py            # API integration tests (FastAPI TestClient)
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_model_training_evaluation.ipynb
│
├── docker/
│   ├── Dockerfile             # Multi-stage production image
│   └── docker-compose.yml     # API + MLflow tracking server
│
├── models/                    # Serialised model artefacts (not committed)
├── mlruns/                    # MLflow experiment tracking (not committed)
└── logs/                      # Application logs (not committed)
```

---

## Data Flow

```
Football-Data.co.uk / StatsBomb
          │
          ▼
  src/data/ingestion.py        ← download_league_season(), load_all_raw()
          │  raw CSV files
          ▼
  src/data/preprocessing.py    ← clean_matches() – rename, cast, encode labels
          │  clean DataFrame
          ▼
  src/features/engineering.py  ← build_features() – Elo, form, rolling averages
          │  feature DataFrame
          ▼
  src/models/train.py          ← prepare_data() → fit models → log to MLflow
          │  best_model.joblib + scaler.joblib
          ▼
  api/main.py (FastAPI)        ← POST /predict → JSON probabilities
```

---

## Machine Learning Models

| Model              | Library      | Role     |
|--------------------|--------------|----------|
| Logistic Regression | scikit-learn | Baseline |
| Random Forest       | scikit-learn | Ensemble |
| XGBoost             | xgboost      | Boosting |
| LightGBM            | lightgbm     | Boosting |

The model with the **lowest test-set log-loss** is saved as `best_model.joblib`.

### Evaluation Metrics

* **Accuracy** – fraction of correct outcome predictions
* **Log-loss** – primary optimisation metric (penalises overconfident wrong predictions)
* **Brier score** – mean squared error of probability estimates (multi-class)
* **Calibration curve** – reliability of probability estimates per class

---

## Feature Engineering

### Elo Rating System
Each team maintains a dynamic Elo rating updated after every match.  
Parameters: `k_factor` (sensitivity), `home_advantage` (offset), `initial_rating`.

### Rolling Form
Points-per-game (3/1/0) over the last `form_window` matches.  
Variants:
- Simple rolling average
- Time-decay weighted (recent matches count more)

### Rolling Averages
Per-team rolling means (window = `form_window`) of:
goals scored/conceded, shots, shots on target, corners.

---

## API Endpoints

| Method | Path             | Description                         |
|--------|------------------|-------------------------------------|
| GET    | `/health`        | API health + model status           |
| POST   | `/predict`       | Single match outcome probabilities  |
| POST   | `/predict/batch` | Batch match predictions             |

### Example request

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "home_team": "Arsenal",
    "away_team": "Chelsea",
    "home_elo": 1650.0,
    "away_elo": 1580.0,
    "home_form": 2.2,
    "away_form": 1.8
  }'
```

### Example response

```json
{
  "home_team": "Arsenal",
  "away_team": "Chelsea",
  "home_win_prob": 0.47,
  "draw_prob": 0.28,
  "away_win_prob": 0.25,
  "predicted_outcome": "Home Win",
  "model_version": "1.0.0"
}
```

---

## Technology Stack

| Layer                | Technology                         | Reason                                          |
|----------------------|------------------------------------|-------------------------------------------------|
| Language             | Python 3.11                        | Standard for ML                                 |
| Data manipulation    | pandas, numpy                      | Tabular data processing                         |
| ML models            | scikit-learn, XGBoost, LightGBM    | Production-proven gradient boosting             |
| Experiment tracking  | MLflow                             | Open-source, easy to self-host                  |
| API                  | FastAPI + Pydantic v2              | Async, auto-docs, type-safe                     |
| Explainability       | SHAP                               | Model-agnostic feature importance               |
| Serialisation        | joblib                             | scikit-learn compatible                         |
| Containerisation     | Docker + Docker Compose            | Portable deployment                             |
| Testing              | pytest + httpx                     | Fast, clean test runner                         |
| Logging              | loguru                             | Structured, easy rotation                       |

---

## Advanced Improvements

1. **Elo rating system** – dynamic team strength tracking across seasons
2. **Time-decay weighting** – exponential decay emphasises recent form
3. **Hyperparameter tuning** – add `sklearn.model_selection.RandomizedSearchCV` or Optuna
4. **Model explainability** – SHAP values via `src/models/evaluate.py::shap_feature_importance`
5. **Calibration** – `calibration_data()` + `sklearn.calibration.CalibratedClassifierCV`
6. **Visualization dashboards** – Plotly/Dash or Streamlit reading from `mlruns/`
7. **Expected Goals (xG)** – when available (StatsBomb data), add as direct feature
8. **Player impact** – aggregate player ratings (FBRef, WhoScored) per lineup
