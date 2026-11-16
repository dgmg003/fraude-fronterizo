from sklearn.model_selection import train_test_split

from frontera_fraud.config import Settings
from frontera_fraud.evaluation.classification import classification_metrics
from frontera_fraud.ingestion.synthetic import generate_synthetic_transactions
from frontera_fraud.services.detection import FraudDetectionService


def test_end_to_end_train_save_load_score(tmp_path):
    cfg = Settings(model_path=tmp_path / "model.joblib", anonymization_salt="test-salt")

    df = generate_synthetic_transactions(n_rows=3000, anomaly_rate=0.03, seed=42)
    train_df, test_df = train_test_split(df, test_size=0.3, random_state=42, stratify=df["is_fraud"])

    FraudDetectionService(cfg).train(train_df).save()
    loaded = FraudDetectionService.load(cfg)
    assert loaded.is_ready

    result = loaded.score(test_df)
    assert len(result) == len(test_df)
    assert result["risk_score"].between(0, 1).all()

    y_pred = (result["risk_level"] == "alto").astype(int).to_numpy()
    metrics = classification_metrics(
        test_df["is_fraud"].to_numpy(), y_pred, result["anomaly_score"].to_numpy()
    )
    # Umbral de humo: si falla, se revisa la causa; no se relaja sin analizarla.
    assert metrics["roc_auc"] > 0.75
