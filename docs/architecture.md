# Architecture

## Overview
The system is split into:
- `frontend/`: React dashboard for prediction, model comparison, what-if simulation, and history.
- `backend/`: FastAPI service for training artifacts, inference, explainability, and persistence.

## Data + Model Layer
1. `backend/app/ml/train.py` downloads the OpenML Mushroom dataset.
2. Categorical traits are one-hot encoded in a scikit-learn pipeline.
3. Five classifiers are trained and evaluated.
4. Artifacts are saved into `backend/saved_models`:
   - model `.joblib` files
   - `metadata.json` (metrics + feature metadata)

## API Layer
- `POST /api/predict`: predicts toxicity from one mushroom profile.
- `POST /api/what-if`: compares two mushroom profiles.
- `GET /api/models/performance`: returns all model metrics.
- `GET /api/features/info`: returns feature descriptions + allowed values.
- `GET /api/history`: returns persisted prediction history.

## Explainability
- Tree models: SHAP contributions.
- Linear models: coefficient-based contributions.
- Non-linear fallback: feature value risk-rate contribution from training metadata.

## Frontend Layer
- Form built from structured feature config (`frontend/lib/patient-config.ts`).
- API calls handled through TanStack Query.
- Results page shows:
  - probability
  - score (0-100)
  - level (Low/Medium/High)
  - top contributing features
  - safety recommendations

## Persistence
- SQLite stores prediction history:
  - model used
  - probability/score/level
  - input payload
  - top factors
