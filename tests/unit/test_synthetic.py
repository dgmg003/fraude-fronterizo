import pandas as pd

from frontera_fraud.ingestion.synthetic import ANOMALY_SCENARIOS, generate_synthetic_transactions

THRESHOLD = 10_000_000.0


def test_generation_is_reproducible():
    a = generate_synthetic_transactions(n_rows=500, seed=1)
    b = generate_synthetic_transactions(n_rows=500, seed=1)
    pd.testing.assert_frame_equal(a, b)


def test_anomaly_count_matches_rate():
    df = generate_synthetic_transactions(n_rows=1000, anomaly_rate=0.05, seed=7)
    assert int(df["is_fraud"].sum()) == 50
    assert set(df.loc[df["is_fraud"] == 1, "scenario"]).issubset(set(ANOMALY_SCENARIOS))


def test_ids_unique_and_amounts_positive():
    df = generate_synthetic_transactions(n_rows=800, seed=2)
    assert df["transaction_id"].is_unique
    assert (df["amount_cop"] > 0).all()


def test_subfacturacion_has_strong_price_deviation():
    df = generate_synthetic_transactions(n_rows=2000, anomaly_rate=0.1, seed=3)
    sub = df[df["scenario"] == "subfacturacion"]
    deviation = sub["unit_price"] / sub["reference_unit_price"] - 1
    assert len(sub) > 0
    assert (deviation < -0.25).all()


def test_fraccionamiento_amounts_fall_below_threshold():
    df = generate_synthetic_transactions(n_rows=2000, anomaly_rate=0.1, seed=4)
    frac = df[df["scenario"] == "fraccionamiento"]
    assert len(frac) > 0
    assert frac["amount_cop"].between(0.90 * THRESHOLD, 0.99 * THRESHOLD).all()


def test_factura_duplicada_reuses_normal_invoices():
    df = generate_synthetic_transactions(n_rows=2000, anomaly_rate=0.1, seed=5)
    normal_invoices = set(df.loc[df["scenario"] == "normal", "invoice_number"])
    dup = df[df["scenario"] == "factura_duplicada"]
    assert len(dup) > 0
    assert dup["invoice_number"].isin(normal_invoices).all()
