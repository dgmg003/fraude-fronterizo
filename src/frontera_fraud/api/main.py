from __future__ import annotations

from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException, Request

from ..config import settings
from ..services.detection import FraudDetectionService
from .schemas import ScoreRequest, ScoreResponse, ScoredTransaction


@asynccontextmanager
async def lifespan(app: FastAPI):
    # El modelo se carga una sola vez al iniciar; no se recarga por solicitud.
    app.state.service = FraudDetectionService.load(settings)
    yield


app = FastAPI(
    title="Detección de fraude en transacciones comerciales fronterizas",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
def health(request: Request) -> dict:
    service: FraudDetectionService = request.app.state.service
    return {"status": "ok", "model_loaded": service.is_ready}


@app.post("/v1/transactions/score", response_model=ScoreResponse)
def score_transactions(payload: ScoreRequest, request: Request) -> ScoreResponse:
    service: FraudDetectionService = request.app.state.service
    if not service.is_ready:
        raise HTTPException(
            status_code=503,
            detail="Modelo no disponible. Ejecute scripts/train_model.py primero.",
        )

    df = pd.DataFrame([t.model_dump() for t in payload.transactions])
    scored = service.score(df)

    results = [
        ScoredTransaction(
            transaction_id=row["transaction_id"],
            anomaly_score=float(row["anomaly_score"]),
            risk_score=float(row["risk_score"]),
            risk_level=row["risk_level"],
            requires_review=row["risk_level"] in ("medio", "alto"),
        )
        for row in scored.to_dict("records")
    ]
    return ScoreResponse(results=results)
