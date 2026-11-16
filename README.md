# Sistema de detección de fraude en transacciones comerciales fronterizas

Prototipo de detección de anomalías para transacciones comerciales fronterizas de Norte de Santander.
Proyecto de la asignatura 167411: Fundamentos de computación paralela y distribuida (Universidad de Pamplona).

> Las alertas son apoyo a la revisión humana, no dictámenes de fraude.

## Estado

- Sprint 1: generador sintético, variables, Isolation Forest, API y pruebas.
- Pendiente: XGBoost, Autoencoder, paralelización (MPI, OpenMP, CUDA) y tablero de alertas.

## Instalación

```bash
python -m venv .venv
source .venv/Scripts/activate   # macOS/Linux: source .venv/bin/activate
pip install -e ".[dev]"
```

## Uso

```bash
python scripts/generate_synthetic_data.py
python scripts/train_model.py
pytest -q
python -m uvicorn frontera_fraud.api.main:app --reload
```

Documentación de la API: http://127.0.0.1:8000/docs

## Datos

El proyecto usa datos sintéticos. Los datos transaccionales reales no se versionan (ver `.gitignore`).
