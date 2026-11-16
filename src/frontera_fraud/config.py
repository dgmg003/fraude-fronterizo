from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    synthetic_data_path: Path = ROOT_DIR / "data" / "synthetic" / "transacciones_sinteticas.csv"
    model_path: Path = ROOT_DIR / "artifacts" / "models" / "isolation_forest.joblib"
    metrics_path: Path = ROOT_DIR / "reports" / "metrics" / "isolation_forest.json"

    random_seed: int = 42
    contamination: float = 0.02
    n_estimators: int = 200

    # Umbral de referencia para el patrón de fraccionamiento.
    # Valor ilustrativo: debe validarse con la normativa vigente antes de usarlo.
    reporting_threshold_cop: float = 10_000_000.0

    # Percentiles de riesgo (0-1) relativos a la distribución de entrenamiento.
    review_risk_threshold: float = 0.95
    alert_risk_threshold: float = 0.98

    anonymization_salt: str = field(
        default_factory=lambda: os.getenv("FRAUDE_SALT", "cambiar-en-produccion")
    )


settings = Settings()
