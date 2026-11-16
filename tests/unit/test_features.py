import pandas as pd
import pytest

from frontera_fraud.features.commercial import FEATURE_COLUMNS, build_features

THRESHOLD = 10_000_000.0


def _row(**overrides):
    row = {
        "transaction_id": "T1",
        "merchant_id": "M1",
        "counterparty_id": "C1",
        "timestamp": pd.Timestamp("2025-01-01"),
        "invoice_number": "F1",
        "product_code": "P1",
        "quantity": 2.0,
        "unit_price": 100_000.0,
        "reference_unit_price": 100_000.0,
        "amount_cop": 200_000.0,
        "exchange_rate_applied": 101.0,
        "exchange_rate_official": 100.0,
        "payer_is_third_party": False,
    }
    row.update(overrides)
    return row


def test_price_deviation_is_relative_to_reference():
    df = pd.DataFrame([_row(unit_price=50_000.0, amount_cop=100_000.0)])
    features = build_features(df, THRESHOLD)
    assert features.loc[0, "price_deviation"] == pytest.approx(-0.5)


def test_exchange_rate_gap():
    df = pd.DataFrame([_row()])
    features = build_features(df, THRESHOLD)
    assert features.loc[0, "exchange_rate_gap"] == pytest.approx(0.01)


def test_near_threshold_flag_boundaries():
    df = pd.DataFrame(
        [
            _row(transaction_id="a", amount_cop=9_500_000.0),   # dentro de [90%, 100%)
            _row(transaction_id="b", amount_cop=10_000_000.0),  # en el umbral: no cuenta
            _row(transaction_id="c", amount_cop=8_000_000.0),   # fuera del rango
        ]
    )
    features = build_features(df, THRESHOLD)
    assert features["near_threshold"].tolist() == [1, 0, 0]


def test_invoice_reused_flags_repeated_invoices():
    df = pd.DataFrame(
        [
            _row(transaction_id="a", invoice_number="F1"),
            _row(transaction_id="b", invoice_number="F1"),
            _row(transaction_id="c", invoice_number="F2"),
        ]
    )
    features = build_features(df, THRESHOLD)
    assert features["invoice_reused"].tolist() == [1, 1, 0]


def test_output_columns_are_stable():
    features = build_features(pd.DataFrame([_row()]), THRESHOLD)
    assert list(features.columns) == FEATURE_COLUMNS
