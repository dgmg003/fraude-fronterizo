from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class Transaction(BaseModel):
    """Transacción comercial fronteriza tal como llega al sistema."""

    transaction_id: str
    merchant_id: str
    counterparty_id: str
    timestamp: datetime
    invoice_number: str
    product_code: str
    quantity: float = Field(gt=0)
    unit_price: float = Field(gt=0)
    reference_unit_price: float = Field(gt=0, description="Precio de referencia de la mercancía")
    amount_cop: float = Field(gt=0)
    exchange_rate_applied: float = Field(gt=0, description="Tasa aplicada en la operación")
    exchange_rate_official: float = Field(gt=0, description="Tasa oficial del día")
    payer_is_third_party: bool = False
