from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "log_amount",
    "quantity",
    "price_deviation",
    "exchange_rate_gap",
    "near_threshold",
    "third_party_payer",
    "invoice_reused",
]


def build_features(df: pd.DataFrame, reporting_threshold: float) -> pd.DataFrame:
    """Construye las variables de entrada del modelo.

    Nota: `invoice_reused` se calcula dentro del lote recibido. En producción
    debe consultarse el histórico de facturas, no solo el lote de la solicitud.
    """
    features = pd.DataFrame(index=df.index)

    features["log_amount"] = np.log1p(df["amount_cop"])
    features["quantity"] = df["quantity"]
    features["price_deviation"] = df["unit_price"] / df["reference_unit_price"] - 1.0
    features["exchange_rate_gap"] = df["exchange_rate_applied"] / df["exchange_rate_official"] - 1.0
    features["near_threshold"] = (
        df["amount_cop"]
        .between(0.90 * reporting_threshold, reporting_threshold, inclusive="left")
        .astype(int)
    )
    features["third_party_payer"] = df["payer_is_third_party"].astype(int)

    invoice_counts = df.groupby("invoice_number")["transaction_id"].transform("size")
    features["invoice_reused"] = (invoice_counts > 1).astype(int)

    return features[FEATURE_COLUMNS]
