from __future__ import annotations

import hashlib
import hmac

import pandas as pd

# Identificadores que se seudonimizan antes de cualquier procesamiento.
# Se usa HMAC para que el mismo valor produzca siempre el mismo seudónimo,
# lo que preserva la detección de facturas repetidas.
PSEUDONYMIZED_COLUMNS = ("merchant_id", "counterparty_id", "invoice_number")


def pseudonymize(value: str, salt: str) -> str:
    digest = hmac.new(salt.encode("utf-8"), str(value).encode("utf-8"), hashlib.sha256)
    return digest.hexdigest()[:16]


def anonymize_frame(
    df: pd.DataFrame,
    salt: str,
    columns: tuple[str, ...] = PSEUDONYMIZED_COLUMNS,
) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        out[col] = out[col].map(lambda v: pseudonymize(v, salt))
    return out
