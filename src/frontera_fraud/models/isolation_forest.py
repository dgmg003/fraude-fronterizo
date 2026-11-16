from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


class IsolationForestDetector:
    """Detector de anomalías no supervisado.

    anomaly_score: mayor valor = más anómalo.
    risk_percentile: posición (0-1) del puntaje frente a la distribución de entrenamiento.
    """

    name = "isolation_forest"

    def __init__(self, contamination: float = 0.02, n_estimators: int = 200, random_state: int = 42):
        self._model = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            random_state=random_state,
        )
        self.feature_names_: list[str] = []
        self._sorted_train_scores: np.ndarray | None = None

    def fit(self, X: pd.DataFrame) -> "IsolationForestDetector":
        self.feature_names_ = list(X.columns)
        self._model.fit(X[self.feature_names_])
        self._sorted_train_scores = np.sort(self.anomaly_score(X))
        return self

    def anomaly_score(self, X: pd.DataFrame) -> np.ndarray:
        return -self._model.score_samples(self._select(X))

    def risk_percentile(self, raw_scores: np.ndarray) -> np.ndarray:
        if self._sorted_train_scores is None:
            raise RuntimeError("El detector no ha sido entrenado.")
        ranks = np.searchsorted(self._sorted_train_scores, raw_scores, side="right")
        return ranks / len(self._sorted_train_scores)

    def _select(self, X: pd.DataFrame) -> pd.DataFrame:
        missing = set(self.feature_names_) - set(X.columns)
        if missing:
            raise ValueError(f"Faltan variables para el modelo: {sorted(missing)}")
        return X[self.feature_names_]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: Path) -> "IsolationForestDetector":
        return joblib.load(path)
