# API Reference

Base URL: `http://localhost:8000/api`

## Health
`GET /health`
- Returns service status, model load state, timestamp.

## Predict
`POST /predict`
- Request body: mushroom feature profile.
- Response: toxicity class, probability, score, level, model name, top factors, recommendations.

## Models Performance
`GET /models/performance`
- Returns metrics for all trained models and selected best model.

## What-If
`POST /what-if`
- Request body:
  - `base_input`
  - `modified_input`
- Response:
  - baseline prediction
  - modified prediction
  - probability and risk deltas

## Feature Metadata
`GET /features/info`
- Returns type/range/description for each model feature.

## History
`GET /history?limit=100`
- Returns recent persisted prediction records sorted by newest first.
