from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..config import Settings, settings as default_settings
from ..features.commercial import build_features
from ..models.isolation_forest import IsolationForestDetector
from ..preprocessing.anonymization import anonymize_frame
from ..scoring.risk_score import classify_risk


class FraudDetectionService:
    """Orquesta anonimización, variables, modelo y calificación de riesgo."""

    def __init__(self, cfg: Settings = default_settings, detector: IsolationForestDetector | None = None):
        self.cfg = cfg
        self.detector = detector

    @property
    def is_ready(self) -> bool:
        return self.detector is not None

    def _prepare(self, df: pd.DataFrame, anonymize: bool = True) -> pd.DataFrame:
        if anonymize:
            df = anonymize_frame(df, self.cfg.anonymization_salt)
        return build_features(df, self.cfg.reporting_threshold_cop)

    def train(self, df: pd.DataFrame, anonymize: bool = True) -> "FraudDetectionService":
        X = self._prepare(df, anonymize)
        self.detector = IsolationForestDetector(
            contamination=self.cfg.contamination,
            n_estimators=self.cfg.n_estimators,
            random_state=self.cfg.random_seed,
        ).fit(X)
        return self

    def score(self, df: pd.DataFrame, anonymize: bool = True) -> pd.DataFrame:
        if self.detector is None:
            raise RuntimeError("El modelo no ha sido entrenado ni cargado.")

        X = self._prepare(df, anonymize)
        raw = self.detector.anomaly_score(X)
        risk = self.detector.risk_percentile(raw)
        levels = classify_risk(
            risk,
            review_threshold=self.cfg.review_risk_threshold,
            alert_threshold=self.cfg.alert_risk_threshold,
        )

        return pd.DataFrame(
            {
                "transaction_id": df["transaction_id"].to_numpy(),
                "anomaly_score": raw,
                "risk_score": risk,
                "risk_level": levels,
            }
        )

    def save(self, path: Path | None = None) -> None:
        if self.detector is None:
            raise RuntimeError("No hay modelo para guardar.")
        self.detector.save(path or self.cfg.model_path)

    @classmethod
    def load(cls, cfg: Settings = default_settings) -> "FraudDetectionService":
        if not cfg.model_path.exists():
            return cls(cfg)
        return cls(cfg, detector=IsolationForestDetector.load(cfg.model_path))
