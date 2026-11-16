from __future__ import annotations

import argparse
from pathlib import Path

from frontera_fraud.config import settings
from frontera_fraud.ingestion.synthetic import generate_synthetic_transactions


def main() -> None:
    parser = argparse.ArgumentParser(description="Genera transacciones sintéticas con anomalías.")
    parser.add_argument("--rows", type=int, default=20_000)
    parser.add_argument("--anomaly-rate", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=settings.random_seed)
    parser.add_argument("--output", type=Path, default=settings.synthetic_data_path)
    args = parser.parse_args()

    df = generate_synthetic_transactions(
        n_rows=args.rows,
        anomaly_rate=args.anomaly_rate,
        seed=args.seed,
        reporting_threshold=settings.reporting_threshold_cop,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)

    print(f"Dataset generado: {args.output} ({len(df)} filas)")
    print(df["scenario"].value_counts().to_string())


if __name__ == "__main__":
    main()
