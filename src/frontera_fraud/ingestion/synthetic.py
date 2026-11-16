from __future__ import annotations

import numpy as np
import pandas as pd

# Tipologías del documento (matriz de tipologías del Objetivo 1).
ANOMALY_SCENARIOS = (
    "subfacturacion",
    "fraccionamiento",
    "factura_duplicada",
    "tercero_pagador",
    "tasa_atipica",
)


def generate_synthetic_transactions(
    n_rows: int = 10_000,
    anomaly_rate: float = 0.03,
    seed: int = 42,
    reporting_threshold: float = 10_000_000.0,
) -> pd.DataFrame:
    """Genera transacciones normales y anomalías inyectadas por tipología.

    Las columnas `scenario` e `is_fraud` son etiquetas de evaluación.
    NO deben usarse como variables de entrada del modelo.
    """
    if not 0 < anomaly_rate < 1:
        raise ValueError("anomaly_rate debe estar entre 0 y 1")

    rng = np.random.default_rng(seed)
    df = _generate_base(rng, n_rows)

    n_anomalies = int(round(n_rows * anomaly_rate))
    anomaly_idx = rng.choice(n_rows, size=n_anomalies, replace=False)

    df["scenario"] = "normal"
    df.loc[anomaly_idx, "scenario"] = rng.choice(ANOMALY_SCENARIOS, size=n_anomalies)

    _apply_anomalies(df, rng, reporting_threshold)

    df["is_fraud"] = (df["scenario"] != "normal").astype(int)
    return df


def _generate_base(rng: np.random.Generator, n: int) -> pd.DataFrame:
    n_products = 50
    product_base_price = rng.uniform(20_000, 500_000, size=n_products)
    product_idx = rng.integers(0, n_products, size=n)

    reference_unit_price = product_base_price[product_idx]
    unit_price = reference_unit_price * np.clip(rng.normal(1.0, 0.05, n), 0.85, 1.15)
    quantity = rng.integers(1, 20, size=n)

    # Tasa oficial ilustrativa; no representa una serie real.
    official = rng.normal(100.0, 2.0, n)
    applied = official * rng.normal(1.0, 0.003, n)

    start = pd.Timestamp("2025-01-01")
    seconds = rng.integers(0, 365 * 24 * 3600, n)

    return pd.DataFrame(
        {
            "transaction_id": [f"T{i:08d}" for i in range(n)],
            "merchant_id": [f"M{x:04d}" for x in rng.integers(1, 201, n)],
            "counterparty_id": [f"C{x:04d}" for x in rng.integers(1, 801, n)],
            "timestamp": (start + pd.to_timedelta(seconds, unit="s")).values,
            "invoice_number": [f"F{i:08d}" for i in range(n)],
            "product_code": [f"P{x:03d}" for x in product_idx],
            "quantity": quantity.astype(float),
            "unit_price": unit_price,
            "reference_unit_price": reference_unit_price,
            "amount_cop": quantity * unit_price,
            "exchange_rate_applied": applied,
            "exchange_rate_official": official,
            "payer_is_third_party": False,
        }
    )


def _apply_anomalies(df: pd.DataFrame, rng: np.random.Generator, threshold: float) -> None:
    normal_invoices = df.loc[df["scenario"] == "normal", "invoice_number"].to_numpy()

    # Subfacturación: precio unitario muy por debajo de la referencia.
    m = df["scenario"] == "subfacturacion"
    k = int(m.sum())
    df.loc[m, "unit_price"] = df.loc[m, "reference_unit_price"].to_numpy() * rng.uniform(0.30, 0.60, k)
    df.loc[m, "amount_cop"] = df.loc[m, "unit_price"].to_numpy() * df.loc[m, "quantity"].to_numpy()

    # Fraccionamiento: monto justo por debajo del umbral.
    m = df["scenario"] == "fraccionamiento"
    k = int(m.sum())
    amount = threshold * rng.uniform(0.90, 0.99, k)
    df.loc[m, "amount_cop"] = amount
    df.loc[m, "unit_price"] = amount / df.loc[m, "quantity"].to_numpy()
    df.loc[m, "reference_unit_price"] = df.loc[m, "unit_price"].to_numpy() * rng.normal(1.0, 0.05, k)

    # Facturación duplicada: reutiliza el número de factura de otra operación.
    m = df["scenario"] == "factura_duplicada"
    df.loc[m, "invoice_number"] = rng.choice(normal_invoices, size=int(m.sum()))

    # Tercero pagador: el pago lo realiza un tercero no relacionado.
    m = df["scenario"] == "tercero_pagador"
    df.loc[m, "payer_is_third_party"] = True

    # Tasa atípica: tasa aplicada muy distinta de la oficial.
    m = df["scenario"] == "tasa_atipica"
    k = int(m.sum())
    factor = np.where(
        rng.random(k) < 0.5,
        rng.uniform(0.70, 0.85, k),
        rng.uniform(1.15, 1.30, k),
    )
    df.loc[m, "exchange_rate_applied"] = df.loc[m, "exchange_rate_official"].to_numpy() * factor
