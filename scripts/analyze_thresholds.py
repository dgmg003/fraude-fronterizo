from __future__ import annotations

from sklearn.model_selection import train_test_split

from frontera_fraud.config import settings
from frontera_fraud.evaluation.thresholds import sweep_thresholds
from frontera_fraud.ingestion.csv_loader import load_transactions
from frontera_fraud.services.detection import FraudDetectionService


def main() -> None:
    service = FraudDetectionService.load(settings)
    if not service.is_ready:
        raise SystemExit("No hay modelo entrenado. Ejecute scripts/train_model.py primero.")

    df = load_transactions(settings.synthetic_data_path)
    # Misma partición que train_model.py, para evaluar sobre el mismo conjunto de prueba.
    _, test_df = train_test_split(
        df,
        test_size=0.3,
        random_state=settings.random_seed,
        stratify=df["is_fraud"],
    )

    result = service.score(test_df)
    table = sweep_thresholds(test_df["is_fraud"].to_numpy(), result["risk_score"].to_numpy())

    out = settings.metrics_path.with_name("threshold_sweep.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)

    print(table.round(3).to_string(index=False))
    print(f"\nTabla guardada en {out}")
    print("Nota: análisis exploratorio sobre el conjunto de prueba; no sirve para fijar el umbral final.")


if __name__ == "__main__":
    main()
