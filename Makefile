.PHONY: install data train api test

install:
	pip install -e ".[dev]"

data:
	python scripts/generate_synthetic_data.py

train:
	python scripts/train_model.py

api:
	uvicorn frontera_fraud.api.main:app --reload

test:
	pytest -q
