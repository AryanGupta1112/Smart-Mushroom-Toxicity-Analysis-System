from __future__ import annotations

import json

from sqlalchemy import desc
from sqlalchemy.orm import Session

from app.models.prediction_history import PredictionHistory
from app.schemas.risk import HistoryItem, HistoryResponse, PredictionResponse


def create_history_record(db: Session, payload: dict, prediction: PredictionResponse) -> PredictionHistory:
    record = PredictionHistory(
        model_name=prediction.model_name,
        prediction=prediction.prediction,
        probability=prediction.probability,
        risk_score=prediction.risk_score,
        risk_level=prediction.risk_level,
        input_payload=json.dumps(payload),
        top_factors=json.dumps([factor.model_dump() for factor in prediction.top_factors]),
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return record


def list_history_records(db: Session, limit: int = 100) -> HistoryResponse:
    rows = (
        db.query(PredictionHistory)
        .order_by(desc(PredictionHistory.timestamp))
        .limit(limit)
        .all()
    )

    items: list[HistoryItem] = []
    for row in rows:
        items.append(
            HistoryItem(
                id=row.id,
                timestamp=row.timestamp,
                model_name=row.model_name,
                prediction=row.prediction,
                probability=row.probability,
                risk_score=row.risk_score,
                risk_level=row.risk_level,
                input_payload=json.loads(row.input_payload),
                top_factors=json.loads(row.top_factors),
            )
        )

    return HistoryResponse(items=items)
