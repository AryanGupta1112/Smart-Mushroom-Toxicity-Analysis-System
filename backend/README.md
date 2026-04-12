# Backend: Smart Mushroom Toxicity Analysis System

## Stack
- FastAPI
- scikit-learn
- SHAP
- SQLAlchemy + SQLite

## Dataset
- Source: OpenML Mushroom dataset (UCI)
- Rows: 8,124
- Features: 22 categorical mushroom traits
- Target: `class` (`p` poisonous, `e` edible)

## What it does
- Trains 5 real ML models:
  - Logistic Regression
  - Decision Tree
  - Random Forest
  - KNN
  - SVM
- Evaluates on held-out test data.
- Saves all model artifacts and metrics to `backend/saved_models`.
- Serves prediction, what-if, explainability, model performance, and history APIs.

## Setup
```bash
cd backend
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
copy .env.example .env
```

## Train Models
```bash
cd backend
set PYTHONPATH=.
python -m app.ml.train
```

## Run API
```bash
cd backend
set PYTHONPATH=.
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## Docker (Backend Only)
```bash
docker compose up --build backend
```

## API Endpoints
- `GET /api/health`
- `POST /api/predict` (optional query: `model_name`)
- `GET /api/models/performance`
- `POST /api/what-if` (optional query: `model_name`)
- `GET /api/features/info`
- `GET /api/history?limit=100`

## Notes
- `INFERENCE_MODEL_NAME` in `.env` can force one specific model.
- If empty, backend uses the best model from training metadata.
- History DB: `backend/data/prediction_history.db`.
