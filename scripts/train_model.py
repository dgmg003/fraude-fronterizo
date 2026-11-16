from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from frontera_fraud.config import settings
from frontera_fraud.evaluation.classification import classification_metrics
from frontera_fraud.ingestion.csv_loader import load_transactions
from frontera_fraud.services.detection import FraudDetectionService


def main() -> None:
    parser = argparse.ArgumentParser(description="Entrena y evalúa el detector Isolation Forest.")
    parser.add_argument("--data", type=Path, default=settings.synthetic_data_path)
    args = parser.parse_args()

    df = load_transactions(args.data)
    labeled = "is_fraud" in df.columns

    if labeled:
        train_df, test_df = train_test_split(
            df,
            test_size=0.3,
            random_state=settings.random_seed,
            stratify=df["is_fraud"],
        )
    else:
        train_df, test_df = df, None

    service = FraudDetectionService(settings).train(train_df)
    service.save()
    print(f"Modelo guardado en {settings.model_path}")

    report: dict = {
        "model": "isolation_forest",
        "n_train": len(train_df),
        "alert_rule": "risk_level == 'alto'",
    }

    if labeled and test_df is not None:
        result = service.score(test_df)
        y_true = test_df["is_fraud"].to_numpy()
        y_pred = (result["risk_level"] == "alto").astype(int).to_numpy()
        report["n_test"] = len(test_df)
        report["metrics"] = classification_metrics(y_true, y_pred, result["anomaly_score"].to_numpy())

        if "scenario" in test_df.columns:
            per_scenario = pd.DataFrame({"scenario": test_df["scenario"].to_numpy(), "detected": y_pred})
            report["recall_by_scenario"] = {
                scenario: float(group["detected"].mean())
                for scenario, group in per_scenario.groupby("scenario")
                if scenario != "normal"
            }

    settings.metrics_path.parent.mkdir(parents=True, exist_ok=True)
    settings.metrics_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
