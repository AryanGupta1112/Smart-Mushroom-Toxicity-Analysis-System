# Smart Mushroom Toxicity Analysis System - Full Project Story

## 1) The Project Story in Simple Words (A to Z)
This project is a full-stack machine learning web application that helps a user check if a mushroom pattern looks likely poisonous.

A user opens the website, selects visible mushroom traits (shape, color, odor, gills, ring, habitat, and more), and submits the form.
The frontend sends those values to the FastAPI backend.
The backend runs a trained ML model, calculates probability and score, builds explanations and warnings, stores the result in history, and sends a structured response back.
The frontend then shows a human-readable result, charts, top factors, recommendations, and comparison tools.

The main purpose is educational decision support for pattern recognition.
It is not a medical or food-safety certification tool.

---

## 2) Main Goal of This Project
The goal is to build a complete production-style ML product with:
- real dataset (large enough),
- real training pipeline,
- multiple models,
- transparent model comparison,
- explainability,
- simple UI for non-developers,
- persistent prediction history,
- what-if simulation,
- and an end-to-end working data flow.

In one sentence: this system turns mushroom observations into an understandable toxicity risk estimate.

---

## 3) What We Predict
For one mushroom profile, the system returns:
- `prediction`: class (0 or 1),
- `probability`: chance that the sample is poisonous,
- `risk_score`: `0-100` (rounded from probability),
- `risk_level`: `Low Risk`, `Medium Risk`, or `High Risk`,
- `top_factors`: strongest features affecting the output,
- `recommendations`: plain-language safety advice,
- `warnings`: reliability warnings when an input value is rare in training data.

Risk mapping logic in `backend/app/utils/risk.py`:
- score `0-39` => `Low Risk`
- score `40-69` => `Medium Risk`
- score `70-100` => `High Risk`

---

## 4) Dataset: What It Is and Why We Chose It
Dataset used:
- OpenML Mushroom dataset (UCI version)
- Rows: `8,124`
- Input features: `22` (all categorical)
- Target: edible vs poisonous

Why this dataset:
- clear structured features,
- enough records (more than 5,000),
- classic ML benchmark,
- reliable for showing full ML product architecture.

In training code, we convert target to numeric:
- poisonous (`p`) => `1`
- edible (`e`) => `0`

---

## 5) Feature Schema (Input Columns)
The model uses 22 fields:
- `cap_shape`
- `cap_surface`
- `cap_color`
- `bruises`
- `odor`
- `gill_attachment`
- `gill_spacing`
- `gill_size`
- `gill_color`
- `stalk_shape`
- `stalk_root`
- `stalk_surface_above_ring`
- `stalk_surface_below_ring`
- `stalk_color_above_ring`
- `stalk_color_below_ring`
- `veil_type`
- `veil_color`
- `ring_number`
- `ring_type`
- `spore_print_color`
- `population`
- `habitat`

The full value map and plain labels are in:
- `backend/app/ml/feature_catalog.py`
- `frontend/lib/patient-config.ts`

---

## 6) Full Model Training Code Explanation (`backend/app/ml/train.py`)
This file is the core training pipeline.

### `RAW_TO_CANONICAL`
A dictionary that renames raw OpenML column names to clean internal names.
Example: `cap-shape` becomes `cap_shape`.

### `DatasetBundle` dataclass
Container used to return:
- features `x` (DataFrame),
- labels `y` (Series),
- source name,
- source row count.

### `_safe_str(series)`
Converts values to trimmed strings.
This makes category handling consistent.

### `load_mushroom_dataset()`
Steps:
1. Fetch data using `fetch_openml(name="mushroom", version=1, as_frame=True)`.
2. Validate that frame exists and is not empty.
3. Rename columns using `RAW_TO_CANONICAL`.
4. Verify all expected features are present.
5. Keep only expected 22 features and convert each to clean strings.
6. Build label `y` by checking if class is `p`.
7. Optional row cap with `TRAINING_MAX_ROWS` env var.
8. Return `DatasetBundle`.

### `_build_preprocessor(scale_numeric)`
Creates a `ColumnTransformer`.

Numeric branch:
- Median imputation,
- Optional standard scaling.

Categorical branch:
- Most frequent imputation,
- One-hot encoding (`handle_unknown="ignore"`).

In this project, all features are categorical, so categorical pipeline is the active branch.

### `_compute_metrics(y_true, y_pred, y_prob)`
Computes:
- accuracy,
- precision,
- recall,
- f1,
- roc_auc.

Returns float values used in metadata and model comparison API.

### `train_and_save_models(output_dir=None)`
This is the full training orchestration function.

It does the following:
1. Create output folder if missing (`saved_models`).
2. Load dataset.
3. Split into train/test with `80/20` and stratification.
4. Define 5 model configurations:
   - Logistic Regression (`max_iter=2500, C=2.0`)
   - Decision Tree (`max_depth=22`)
   - Random Forest (`n_estimators=450, max_depth=40`)
   - KNN (`n_neighbors=5, weights="distance"`)
   - SVM (`RBF kernel, C=3.0, probability=True`)
5. For each model:
   - Build preprocessing + estimator pipeline,
   - Fit on training data,
   - Predict class and probability on test data,
   - Compute metrics,
   - Save model as `.joblib`.
6. Pick best model by highest accuracy.
7. Build metadata with:
   - dataset stats,
   - metric table,
   - feature stats,
   - per-value target rates,
   - per-value frequencies.
8. Save metadata to `saved_models/metadata.json`.
9. Return metadata.

### `if __name__ == "__main__":`
Running `python -m app.ml.train` executes training and prints metadata.

---

## 7) Validation: What We Currently Use
Current official training validation is:
- one stratified holdout split (`80% train / 20% test`),
- standard classification metrics.

Important interpretation:
- This dataset is highly separable, so very high and sometimes perfect scores can happen.
- This does not mean "perfect in all real-world conditions".
- It means "very strong on this dataset and split style".

---

## 8) Full Inference Code Explanation (`backend/app/services/model_service.py`)
`ModelService` handles model loading, prediction, explanations, and reliability warnings.

### `ensure_ready()`
- Checks if metadata exists.
- If missing and `AUTO_TRAIN_MODELS=true`, triggers training automatically.
- Loads metadata and all model `.joblib` files.
- Sets `ready=True`.

### `available_model_metrics()`
Converts metadata metric entries into typed response objects.

### Explainability helpers
Three explanation modes are implemented:

1. Tree SHAP (`_explain_with_shap`)
- For tree models (`random_forest`, `decision_tree`), tries SHAP.
- Gets transformed features, SHAP values, and groups one-hot columns back to original feature names.
- Returns top 5 factors by absolute impact.

2. Coefficient/importance (`_explain_with_coefficients`)
- For models exposing coefficients or feature importances.
- Computes contribution from transformed values.
- Groups by original feature and returns top 5.

3. Value-rate fallback (`_explain_from_value_rates`)
- If SHAP/coeff fails, uses metadata frequency statistics.
- Compares selected value risk rate to global positive rate.
- Returns top 5 by strongest delta.

### `explain(model_name, row_df)`
Routing logic:
- Tree model => try SHAP first.
- Else try coefficient/importance.
- If anything fails => value-rate fallback.

### `_resolve_model_name()`
Model selection priority:
1. request query param `model_name`,
2. env var `INFERENCE_MODEL_NAME`,
3. metadata `best_model`,
4. `DEFAULT_MODEL_NAME`.

### `_build_reliability_warnings(payload)`
Checks how common each chosen value is in training data.
If value frequency is <= 1%, adds a warning message.

### `predict(payload, model_name=None)`
End-to-end prediction path:
1. Ensure models loaded.
2. Resolve model name.
3. Convert payload to one-row DataFrame.
4. Get probability from `predict_proba`.
5. Convert to class, risk score, and risk level.
6. Build top factors via explainability.
7. Build recommendations via rule engine.
8. Build reliability warnings.
9. Return `PredictionResponse`.

---

## 9) Recommendation Engine (`backend/app/services/recommendation_service.py`)
This is a rule-based layer added on top of ML output.

Examples of rules:
- dangerous odors (`c, f, m, p, s, y`) => strong unsafe warning,
- green spore print (`r`) => strong warning,
- narrow gills (`n`) => caution,
- certain ring types (`e`, `n`) => caution,
- uncertain habitats (`u`, `w`) => caution.

Then it adds a probability-based summary:
- `>= 0.7` high toxicity confidence warning,
- `<= 0.3` lower-risk reminder with expert-check caution,
- otherwise "uncertain zone" guidance.

---

## 10) API Layer (`backend/app/api/routes.py`)
Main routes:

- `GET /api/health`
  - returns backend and model-ready status.

- `POST /api/predict`
  - input: one mushroom profile,
  - output: prediction + score + explanations + recommendations,
  - also saves to history DB.

- `GET /api/models/performance`
  - returns all model metrics + best model.

- `GET /api/models/training-summary`
  - returns source, rows, feature counts, split style, trained models, active inference model.

- `POST /api/what-if`
  - input: `base_input` and `modified_input`,
  - returns both predictions and deltas.

- `GET /api/features/info`
  - returns feature metadata and allowed values.

- `GET /api/history`
  - returns recent saved prediction records.

---

## 11) Validation and Input Safety (`backend/app/schemas/risk.py`)
`PatientInput` uses strict allowed-value validation for each feature code.
If an unknown code is sent, the API rejects it.

This protects model integrity by preventing invalid category inputs.

---

## 12) Database and History
SQLite is used for prediction history.

Files:
- `backend/app/core/database.py`
- `backend/app/models/prediction_history.py`
- `backend/app/services/history_service.py`

Each prediction saves:
- timestamp,
- model name,
- class,
- probability,
- score,
- level,
- full input payload,
- top factors.

This supports trend charts and audit-style review in the frontend.

---

## 13) Backend Startup Flow (`backend/app/main.py`)
When FastAPI starts:
1. ensure `data/` and `saved_models/` folders exist,
2. create DB tables,
3. load model artifacts (or auto-train if enabled),
4. enable CORS,
5. serve API under `/api`.

---

## 14) Frontend Architecture and Flow
Frontend uses Vite + React + TypeScript.

Routing entry:
- `frontend/src/main.tsx`
- `frontend/src/App.tsx`

Shared shell/navigation:
- `frontend/components/app-shell.tsx`

Provider setup:
- `frontend/app/providers.tsx`
  - React Query for API state,
  - theme provider (light/dark),
  - tooltip provider.

HTTP client:
- `frontend/services/api.ts`
  - Axios with `VITE_API_BASE_URL`.

Data hooks:
- `frontend/hooks/use-risk-api.ts`
  - queries: health, model metrics, training summary, features, history,
  - mutations: predict and what-if.

---

## 15) Frontend Pages and What Each One Does

### Home (`frontend/app/page.tsx`)
- Shows system status, dataset size, best model, training timestamp.
- Shows simple pipeline and model summary table.

### Check Mushroom (`frontend/app/dashboard/page.tsx`)
- Main prediction form.
- Model selector (user can choose model at runtime).
- "Generate Description" button turns selected values into a plain paragraph.
- Submit button calls `/predict`.
- Displays score gauge, level, warnings, top factors chart, recommendations.

### Model Scores (`frontend/app/models/page.tsx`)
- Fetches `/models/performance` and training summary.
- Ranks models.
- Shows "best model for which goal" guidance.
- Includes chart and detailed table.

### Compare Two Cases (`frontend/app/what-if/page.tsx`)
- Two side-by-side forms (Mushroom A and B).
- Both support paragraph generation from selected traits.
- Calls `/what-if` and shows score/probability deltas.

### Prediction History (`frontend/app/history/page.tsx`)
- Shows saved past predictions from SQLite history.
- Includes trend chart and table.

### About (`frontend/app/about/page.tsx`)
- Explains pipeline and live training details in non-technical style.

---

## 16) End-to-End Pipeline (From User Click to Final Result)
1. User opens dashboard.
2. Frontend loads model/feature info from backend.
3. User selects mushroom features.
4. User can generate a human-readable paragraph of the selected profile.
5. User clicks "Check Safety".
6. Frontend sends JSON to `POST /api/predict`.
7. Backend validates input codes with Pydantic.
8. Backend chooses model (query param or configured default logic).
9. Pipeline preprocesses values (one-hot encoding).
10. Model predicts probability and class.
11. Backend builds score and risk level.
12. Backend computes explainability top factors.
13. Backend adds rule-based recommendations.
14. Backend adds reliability warnings for rare values.
15. Backend writes record into SQLite history.
16. Backend returns full response JSON.
17. Frontend renders cards, charts, factor table, and recommendations.
18. History page updates with the new record.

---

## 17) Environment Variables You Can Control
In `backend/.env` or `.env.example`:
- `AUTO_TRAIN_MODELS=true|false`
  - `true`: auto-train if artifacts missing.
  - `false`: require manual training first.

- `INFERENCE_MODEL_NAME=`
  - force one model for inference if set (for example `random_forest`).

- `DATASET_PROFILE=mushroom`
  - dataset profile marker.

- `TRAINING_MAX_ROWS=0`
  - `0` means full dataset.
  - positive number means sample down to that many rows.

Frontend:
- `VITE_API_BASE_URL=http://localhost:8000/api`

---

## 18) How to Run the Full System

### Backend
1. `cd backend`
2. `python -m venv .venv`
3. `.venv\\Scripts\\activate`
4. `pip install -r requirements.txt`
5. `copy .env.example .env`
6. `set PYTHONPATH=.`
7. `python -m app.ml.train`
8. `uvicorn app.main:app --reload --host 0.0.0.0 --port 8000`

### Frontend
1. `cd frontend`
2. `npm install`
3. set `VITE_API_BASE_URL` in `.env.local`
4. `npm run dev`

---

## 19) Docker Notes
`docker-compose.yml` currently has backend as default service.
Frontend is behind a compose profile (`frontend`) and can be started when needed.
Model artifacts are mounted from host, so training can be run separately outside Docker.

---

## 20) Why Accuracy Can Be Very High Here
This dataset has strong separability across feature combinations.
With random stratified splits, models can reach very high scores.
This is normal for this benchmark dataset.

Important boundary:
- high benchmark score does not equal guaranteed real-world safety.
- users should never rely only on model output for edibility decisions.

---

## 21) Current Strengths
- Real end-to-end ML product, not a toy script.
- Multiple models with saved artifacts and live switching.
- Explainability and safety warnings.
- Historical logging and charts.
- Plain-language UI for new users.
- What-if simulation for scenario analysis.

---

## 22) Current Limitations and Next Technical Improvements
Limitations:
- primary validation is one holdout split,
- dataset is controlled benchmark data,
- no external field-image or lab data integration.

Good next upgrades:
- stratified k-fold cross-validation report,
- out-of-distribution evaluation with grouped splits,
- calibration curves and threshold tuning,
- confidence intervals for metrics,
- external dataset benchmark.

---

## 23) Final Summary
This project is a complete full-stack ML system that starts with human observations and ends with an explainable toxicity risk estimate.
The backend is modular, the frontend is user-friendly, the pipeline is real, and the whole system is connected from data training to live prediction history.

In simple terms: we built a practical ML product, not just a notebook experiment.
