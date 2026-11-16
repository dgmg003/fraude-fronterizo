import numpy as np

from frontera_fraud.evaluation.thresholds import sweep_thresholds


def test_sweep_counts_alerts_per_threshold():
    y = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    risk = np.array([0.99, 0.96, 0.5, 0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8])
    table = sweep_thresholds(y, risk, percentiles=(0.95, 0.98))
    assert table["alertas"].tolist() == [2, 1]
    assert table["recall"].tolist() == [1.0, 1.0]
    assert table["precision"].tolist() == [0.5, 1.0]
