import numpy as np

from frontera_fraud.scoring.risk_score import classify_risk


def test_classify_risk_levels_by_thresholds():
    risk = np.array([0.10, 0.96, 0.99])
    levels = classify_risk(risk, review_threshold=0.95, alert_threshold=0.98).tolist()
    assert levels == ["bajo", "medio", "alto"]
