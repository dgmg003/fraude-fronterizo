import pytest

from frontera_fraud.evaluation.classification import classification_metrics
from frontera_fraud.evaluation.performance import amdahl_speedup, efficiency, speedup


def test_classification_metrics_on_known_example():
    y_true = [1, 1, 0, 0, 0]
    y_pred = [1, 0, 1, 0, 0]
    m = classification_metrics(y_true, y_pred)
    assert m["precision"] == pytest.approx(0.5)
    assert m["recall"] == pytest.approx(0.5)
    assert m["false_positive_rate"] == pytest.approx(1 / 3)
    assert m["specificity"] == pytest.approx(2 / 3)
    assert m["f1"] == pytest.approx(0.5)


def test_speedup_and_efficiency():
    assert speedup(10.0, 2.0) == pytest.approx(5.0)
    assert efficiency(10.0, 2.0, 5) == pytest.approx(1.0)


def test_amdahl_limit():
    assert amdahl_speedup(0.9, 10) == pytest.approx(1 / 0.19)