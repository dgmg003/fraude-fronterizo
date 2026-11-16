import pandas as pd

from frontera_fraud.preprocessing.anonymization import anonymize_frame, pseudonymize


def test_pseudonym_is_deterministic_and_short():
    assert pseudonymize("M0001", "salt") == pseudonymize("M0001", "salt")
    assert len(pseudonymize("M0001", "salt")) == 16


def test_different_salt_produces_different_pseudonym():
    assert pseudonymize("M0001", "salt-a") != pseudonymize("M0001", "salt-b")


def test_same_invoice_keeps_same_pseudonym():
    df = pd.DataFrame(
        {
            "merchant_id": ["M1", "M2"],
            "counterparty_id": ["C1", "C2"],
            "invoice_number": ["F1", "F1"],
            "amount_cop": [1.0, 2.0],
        }
    )
    out = anonymize_frame(df, "salt")
    assert out["invoice_number"].iloc[0] == out["invoice_number"].iloc[1]
    assert out["amount_cop"].tolist() == [1.0, 2.0]
    assert out["merchant_id"].iloc[0] != "M1"
