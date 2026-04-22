# Football Match Prediction

> **Production-grade machine learning project** for predicting football (soccer) match outcomes – Win / Draw / Loss – with calibrated probabilities.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688)](https://fastapi.tiangolo.com/)
[![MLflow](https://img.shields.io/badge/tracking-MLflow-0194E2)](https://mlflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Overview

This project implements an end-to-end ML pipeline to predict the outcome of football matches using historical match statistics, Elo ratings, and team form metrics. The trained model is served through a REST API built with FastAPI.

**Predicted outcomes:**
- 🏠 Home Win probability
- 🤝 Draw probability  
- ✈️ Away Win probability

---

## Project Structure

```
football-match-prediction/
├── configs/config.yaml          # Central configuration
├── src/
│   ├── data/                    # Data ingestion, preprocessing, pipeline
│   ├── features/                # Feature engineering (Elo, form, rolling stats)
│   ├── models/                  # Train, evaluate, predict
│   └── utils/                   # Helpers (config, logging, seeds)
├── api/                         # FastAPI REST API
├── tests/                       # pytest test suite
├── notebooks/                   # Exploratory & training notebooks
├── docker/                      # Dockerfile + docker-compose
└── docs/architecture.md         # Detailed architecture documentation
```

See [`docs/architecture.md`](docs/architecture.md) for a full description.

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
pip install -e .
```

### 2. Download data

```bash
python -m src.data.pipeline
```

This downloads match data from [Football-Data.co.uk](https://www.football-data.co.uk/) for the leagues and seasons configured in `configs/config.yaml`, preprocesses it, and builds the feature dataset.

### 3. Train models

```bash
python -m src.models.train
# or
make train
```

All models (Logistic Regression, Random Forest, XGBoost, LightGBM) are trained, compared and logged to **MLflow**. The best model is saved to `models/best_model.joblib`.

View results:
```bash
mlflow ui --backend-store-uri mlruns
```

### 4. Run the API

```bash
make run-api
# or
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

Interactive docs: http://localhost:8000/docs

### 5. Make a prediction

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

---

## Docker

```bash
# Build and start API + MLflow tracking server
make docker-up

# Or manually:
docker-compose -f docker/docker-compose.yml up --build
```

---

## Testing

```bash
make test
# With coverage:
make test-cov
```

---

## Data Sources

| Source | Description | Access |
|--------|-------------|--------|
| [Football-Data.co.uk](https://www.football-data.co.uk/) | Historical match results, odds, stats | Free CSV download |
| [StatsBomb Open Data](https://github.com/statsbomb/open-data) | Detailed event & tracking data | Free via GitHub |
| [Kaggle Datasets](https://www.kaggle.com/datasets?search=football) | Community-curated football datasets | Free (account required) |

Configured leagues: Premier League, La Liga, Bundesliga, Serie A, Ligue 1.

---

## Feature Engineering

| Feature | Description |
|---------|-------------|
| `home_elo` / `away_elo` | Dynamic Elo rating (updated after every match) |
| `elo_diff` | Elo rating difference (home − away) |
| `home_form` / `away_form` | Points per game over last N matches |
| `home_form_decayed` | Time-decay weighted form (recency bias) |
| `*_goals_scored_avg` | Rolling average goals scored (N matches) |
| `*_goals_conceded_avg` | Rolling average goals conceded (N matches) |
| `*_shots_avg` | Rolling average shots |
| `*_shots_on_target_avg` | Rolling average shots on target |
| `*_corners_avg` | Rolling average corners |

---

## Models & Metrics

| Model | Accuracy | Log-loss | Brier Score |
|-------|----------|----------|-------------|
| Logistic Regression | ~52% | ~1.00 | ~0.64 |
| Random Forest | ~54% | ~0.99 | ~0.62 |
| XGBoost | ~55% | ~0.97 | ~0.61 |
| LightGBM | ~55% | ~0.96 | ~0.60 |

*Approximate values on Premier League 2021–2024 data.*

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check + model status |
| `POST` | `/predict` | Predict single match outcome |
| `POST` | `/predict/batch` | Predict multiple matches |

---

## Technology Stack

- **ML**: scikit-learn, XGBoost, LightGBM
- **API**: FastAPI + Pydantic v2 + Uvicorn
- **Experiment tracking**: MLflow
- **Explainability**: SHAP
- **Containerisation**: Docker + Docker Compose
- **Testing**: pytest + httpx

---

## Advanced Features

- 🎯 **Elo rating system** with configurable K-factor and home advantage
- ⏱️ **Time-decay weighted form** (exponential decay, configurable half-life)
- 📊 **SHAP feature importance** for model explainability
- 🔬 **Calibration analysis** for probability reliability
- 🧪 **MLflow experiment tracking** with cross-validation metrics

---

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Run tests: `make test`
4. Submit a pull request

---

## License

MIT License – see [LICENSE](LICENSE) for details.
