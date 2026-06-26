# ML Playground API

[![CI](https://github.com/yourusername/ml-playground-api/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/yourusername/ml-playground-api/actions/workflows/ci-cd.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A lightweight FastAPI + scikit-learn service for training and serving classification models on built-in datasets. No deep learning dependencies required.

## Features

- Built-in datasets: iris, wine, breast_cancer, digits
- Train models: Random Forest, SVM, Logistic Regression
- Model registry with unique IDs per trained model
- Prediction endpoint with probability support
- Feature count validation
- OpenAPI docs at `/docs`

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Service health and model count |
| GET | `/datasets` | List available datasets with feature info |
| POST | `/train` | Train a model on a dataset |
| GET | `/models` | List all trained models |
| GET | `/models/{id}` | Get details for a specific model |
| POST | `/predict` | Run inference with a trained model |

## Quick Start

```bash
# Install
pip install -r requirements.txt

# Run
uvicorn src.main:app --reload

# OpenAPI docs at http://localhost:8000/docs
```

### Examples

```bash
# Health check
curl http://localhost:8000/health

# List datasets
curl http://localhost:8000/datasets | jq

# Train a Random Forest on iris
curl -X POST http://localhost:8000/train \
  -H "Content-Type: application/json" \
  -d '{"dataset_name": "iris", "model_type": "random_forest", "params": {"random_state": 42}}'

# Train an SVM on wine
curl -X POST http://localhost:8000/train \
  -H "Content-Type: application/json" \
  -d '{"dataset_name": "wine", "model_type": "svm"}'

# List trained models
curl http://localhost:8000/models | jq

# Get model details
curl http://localhost:8000/models/<MODEL_ID> | jq

# Predict with a model
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"model_id": "<MODEL_ID>", "features": [5.1, 3.5, 1.4, 0.2]}'
```

## Development

```bash
# Install dependencies
pip install -r requirements.txt

# Lint
ruff check src/ tests/

# Test
pytest tests/ -v --cov=src
```

## Project Structure

```
src/
├── __init__.py
├── main.py           # FastAPI app and route definitions
├── core/
│   ├── __init__.py
│   ├── datasets.py   # Dataset loaders
│   └── trainer.py    # Model training logic
├── models/
│   ├── __init__.py
│   ├── schemas.py    # Pydantic request/response models
│   └── registry.py   # In-memory model registry
tests/
├── test_datasets.py
├── test_trainer.py
├── test_registry.py
└── test_api.py
```

## License

MIT
