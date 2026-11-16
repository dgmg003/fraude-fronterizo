from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = [
    "transaction_id",
    "merchant_id",
    "counterparty_id",
    "timestamp",
    "invoice_number",
    "product_code",
    "quantity",
    "unit_price",
    "reference_unit_price",
    "amount_cop",
    "exchange_rate_applied",
    "exchange_rate_official",
    "payer_is_third_party",
]

POSITIVE_COLUMNS = [
    "quantity",
    "unit_price",
    "reference_unit_price",
    "amount_cop",
    "exchange_rate_applied",
    "exchange_rate_official",
]


def load_transactions(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["timestamp"])
    return validate_transactions(df)


def validate_transactions(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas obligatorias: {missing}")

    before = len(df)
    df = df.dropna(subset=REQUIRED_COLUMNS)
    df = df.drop_duplicates(subset=["transaction_id"], keep="first")
    df = df[(df[POSITIVE_COLUMNS] > 0).all(axis=1)]

    dropped = before - len(df)
    if dropped:
        logger.warning("Se descartaron %d filas inválidas o duplicadas", dropped)
    return df.reset_index(drop=True)
