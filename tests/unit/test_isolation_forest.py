import numpy as np
import pandas as pd
import pytest

from frontera_fraud.models.isolation_forest import IsolationForestDetector


def _normal_data(seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"a": rng.normal(size=500), "b": rng.normal(size=500)})


def _fitted_detector(X: pd.DataFrame) -> IsolationForestDetector:
    return IsolationForestDetector(contamination=0.02, n_estimators=100, random_state=42).fit(X)


def test_outlier_gets_higher_score_than_inlier():
    det = _fitted_detector(_normal_data())
    outlier = pd.DataFrame({"a": [10.0], "b": [10.0]})
    inlier = pd.DataFrame({"a": [0.0], "b": [0.0]})
    assert det.anomaly_score(outlier)[0] > det.anomaly_score(inlier)[0]


def test_risk_percentile_is_in_unit_interval():
    X = _normal_data()
    det = _fitted_detector(X)
    p = det.risk_percentile(det.anomaly_score(X))
    assert ((p > 0) & (p <= 1)).all()


def test_unfitted_detector_raises():
    det = IsolationForestDetector()
    with pytest.raises(RuntimeError):
        det.risk_percentile(np.array([0.5]))
