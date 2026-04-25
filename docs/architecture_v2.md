# Football Match Prediction – Architecture v2

## Overview

This document describes the enhanced sports intelligence platform architecture, building on the v1 baseline to incorporate advanced feature engineering, ensemble modeling, calibration evaluation, and contextual intelligence.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Data Pipeline                                 │
│  Football-Data.co.uk → ingestion → preprocessing → features    │
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│                 Feature Engineering (4 Tiers)                    │
│                                                                  │
│  Tier 1 (Core):      Elo, form, rolling averages                │
│  Tier 2 (Advanced):  Opponent-adjusted, tactical, weighted form │
│  Tier 3 (Interaction): pressure×form, attack×defence, elo×imp  │
│  Tier 4 (Temporal):  Lag features, EWMA, momentum indicators   │
│                                                                  │
│  + Match Importance: Pressure Index per team per match          │
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│                 Ensemble Modeling                                 │
│                                                                  │
│  Level 0 (Base Models):                                         │
│    ├── Random Forest                                             │
│    ├── XGBoost                                                   │
│    ├── LightGBM                                                  │
│    ├── Neural Network (MLP / LSTM)                              │
│    └── Bayesian Ridge (uncertainty-aware)                       │
│                                                                  │
│  Level 1 (Meta-Learner):                                        │
│    └── Logistic Regression (trained on OOF predictions)        │
│        + Platt calibration for probability reliability           │
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│                 Evaluation Framework                             │
│                                                                  │
│  • Calibration: Brier Score, ECE, Reliability Diagrams          │
│  • Temporal: Rolling window backtesting (walk-forward)          │
│  • Explainability: SHAP feature importance                      │
│  • Stratified: top-4 vs relegation zone vs mid-table           │
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│                 Production API (FastAPI)                          │
│                                                                  │
│  POST /predict          – single match prediction               │
│  POST /predict/batch    – batch predictions                     │
│  GET  /health           – service health check                  │
│                                                                  │
│  Input validation via Pydantic (MatchFeaturesRequest)           │
│  Calibrated output: {home_win_prob, draw_prob, away_win_prob}   │
└─────────────────────────────────────────────────────────────────┘
```

---

## Feature Engineering

### Match Importance (Pressure Index)

Each team is assigned a Pressure Index ∈ [0, 1] for each match:

```
pressure_index = max(title_pressure, top4_pressure, relegation_pressure)
                 × season_stage_weight
```

**Inputs:**
- Current league standings (points, position)
- Remaining matches in the season
- Distance to key thresholds (title: ±5 pts, top-4: ±3 pts, relegation: ±3 pts)

**Season stage weight:**
- Weeks 1–5: 0.40 (early, standings uninformative)
- Weeks 6–10: 0.50–0.75 (progressive increase)
- Weeks 11+: 0.75–1.0 (linear ramp to final matchday)

### Feature Hierarchy

| Tier | Features | Source |
|------|----------|--------|
| 1 – Core | Elo ratings, basic form, rolling averages | `engineering.py` |
| 2 – Advanced | Opponent-adjusted goals, tactical proxies, weighted form | `advanced_features.py` |
| 3 – Interaction | pressure×form, attack/defence mismatch, elo×importance | `advanced_features.py` |
| 4 – Temporal | Lag features (t-1, t-2, t-3), EWMA, win streaks | `temporal_features.py` |
| + Importance | Pressure Index home/away | `match_importance.py` |

---

## Ensemble Architecture

### Level 0: Base Models

| Model | Strengths |
|-------|-----------|
| Random Forest | Robust to outliers, handles non-linear interactions |
| XGBoost | High accuracy, built-in regularisation |
| LightGBM | Fast training, handles large feature sets |
| MLP Neural Network | Captures non-linear combinations of features |
| Bayesian Ridge | Uncertainty quantification, well-calibrated |
| Logistic Regression | Interpretable baseline, fast |

### Level 1: Meta-Learner

A Logistic Regression is trained on **out-of-fold (OOF)** predictions from all base models. This learns optimal blending weights and can capture which base model performs best in different match contexts.

Platt calibration (sigmoid) is applied to the meta-learner to ensure predicted probabilities are reliable.

### Training Strategy

1. Split data temporally: train / validation / test
2. Generate OOF predictions using k-fold cross-validation on training set
3. Train meta-learner on OOF predictions
4. Re-train base models on full training + validation data
5. Evaluate on held-out test set

---

## Evaluation

### Calibration Metrics

| Metric | Target | Description |
|--------|--------|-------------|
| Brier Score | < 0.58 | Mean squared probability error |
| Mean ECE | < 0.08 | Average confidence-accuracy gap per bin |

### Rolling Window Backtest

Simulates live deployment:
1. Train on data up to date T
2. Evaluate predictions for window T → T+N
3. Slide window forward by step size
4. Report per-fold and aggregate metrics

### SHAP Explainability

Per-prediction feature importance via SHAP values, using:
- `TreeExplainer` for tree-based models (fast)
- `KernelExplainer` fallback for other models

---

## Module Reference

```
src/
├── features/
│   ├── engineering.py          # Core: Elo, form, rolling stats
│   ├── match_importance.py     # Pressure Index calculation
│   ├── advanced_features.py    # Opponent-adjusted, tactical, interaction
│   └── temporal_features.py    # Lag, EWMA, momentum, sequence prep
├── models/
│   ├── train.py                # Training pipeline entry point
│   ├── ensemble.py             # Stacking/blending ensemble
│   ├── neural_network.py       # MLP and Keras feedforward nets
│   ├── bayesian.py             # Bayesian Ridge with uncertainty
│   ├── sequential.py           # LSTM and SimpleRNN sequence models
│   ├── evaluate.py             # Evaluation utilities
│   └── predict.py              # Inference / predictor class
├── evaluation/
│   ├── calibration.py          # Brier, ECE, reliability diagrams
│   ├── rolling_backtest.py     # Walk-forward backtesting
│   └── explainability.py       # SHAP-based feature importance
└── data/
    ├── ingestion.py            # Data download (Football-Data.co.uk)
    └── preprocessing.py        # Cleaning and encoding

configs/
├── config.yaml                 # Main configuration
├── features_config.yaml        # Feature engineering settings
└── model_config.yaml           # Model architecture and hyperparams

api/
├── main.py                     # FastAPI application
└── schemas.py                  # Pydantic request/response schemas
```

---

## Configuration

All behaviour is controlled via YAML configuration files under `configs/`:

- **`config.yaml`** – global settings (data sources, seeds, API)
- **`features_config.yaml`** – feature engineering pipeline settings
- **`model_config.yaml`** – model architecture and hyperparameters

---

## Expected Outcomes

| Metric | v1 Baseline | v2 Target |
|--------|-------------|-----------|
| Accuracy | ~55% | 58–62% |
| Log-Loss | ~0.96 | 0.92–0.94 |
| Brier Score | ~0.60 | < 0.58 |
| ECE | N/A | < 0.08 |
| Explainability | None | SHAP per match |

---

## Dependencies

Key additions in v2:
- `shap` – SHAP explainability
- `rapidfuzz` – fuzzy team name matching in live features
- `mlflow` – experiment tracking
- `optuna` (optional) – Bayesian hyperparameter optimisation
- `tensorflow` (optional) – deep learning models
