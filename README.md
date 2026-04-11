# Smart Mushroom Toxicity Analysis System

Full-stack ML system for high-accuracy mushroom toxicity prediction using a large real dataset (8,124 rows).

## Project Goal
This project predicts whether a mushroom profile is likely poisonous or edible, explains the top factors behind the result, compares multiple ML models, and supports what-if simulation.

## Dataset
- Source: OpenML Mushroom dataset (UCI version)
- Records: 8,124 (meets 5,000+ requirement)
- Input features: 22 categorical mushroom traits
- Target label: poisonous (`p`) vs edible (`e`)

## Model Pipeline
1. Download and normalize dataset from OpenML.
2. One-hot encode categorical features.
3. Split into train/test (`80/20`, stratified).
4. Train 5 models:
   - Logistic Regression
   - Decision Tree
   - Random Forest
   - KNN
   - SVM
5. Evaluate with accuracy, precision, recall, F1, ROC-AUC.
6. Save trained models and metadata in `backend/saved_models`.

## Accuracy
After re-training on this dataset, all five models currently score near-perfect on the held-out test split for this dataset version.

## Architecture
- Frontend: Vite + React + TypeScript + shadcn/ui
- Backend: FastAPI + scikit-learn + SHAP + SQLite
- Data flow: Frontend -> API -> selected model -> explainability + recommendations -> UI

## Folder Structure
```text
project-root/
  backend/
    app/
      api/
      core/
      models/
      schemas/
      services/
      ml/
      utils/
      main.py
    data/
    notebooks/
    saved_models/
    requirements.txt
    README.md

  frontend/
    app/
    components/
    components/ui/
    lib/
    hooks/
    services/
    types/
    public/
    styles/
    README.md
    package.json

  docs/
  docker-compose.yml
  README.md
```

## Quick Start

### 1) Backend
```bash
cd backend
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
copy .env.example .env
set PYTHONPATH=.
python -m app.ml.train
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 2) Frontend
```bash
cd frontend
npm install
```
Create `frontend/.env.local`:
```bash
VITE_API_BASE_URL=http://localhost:8000/api
```
Run:
```bash
npm run dev
```

## Docker
Backend only:
```bash
docker compose up --build backend
```

## Core APIs
- `POST /api/predict`
- `GET /api/models/performance`
- `POST /api/what-if`
- `GET /api/features/info`
- `GET /api/history`
- `GET /api/health`
