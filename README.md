[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)]
[![MLOps](https://img.shields.io/badge/MLOps-Regression-success)]
[![Docker](https://img.shields.io/badge/Docker-API--only-blue)]
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)]

# Health Insurance Cost Prediction MLOps Project

Authors: Eduardo Debes, Nicolas Guadamillas, Benedetta Pagliardi, Marco Siñaniz, Max Vanderlinden  
Course context: MLOps: Master in Business Analytics and Data Science  
Repository: `marcosinaniz-ai/mlops_group_1`  
Branch: `dockerize`

## Overview

This repository turns a notebook-based machine learning project into a modular, testable, and deployable MLOps system for **health insurance cost prediction**.

The project predicts individual insurance charges from structured demographic and lifestyle variables such as age, sex, BMI, number of children, smoking status, and region.

It includes:

- a modular `src/` pipeline
- centralized configuration in `config.yaml`
- unit tests with `pytest`
- experiment tracking and artifact logging with Weights & Biases
- a FastAPI inference service
- Docker packaging for serving the API
- a serialized model artifact for consistent inference

This repository is designed as an MLOps project that separates experimentation from production code and provides a clean path from training to serving.

## Business objective

The goal is to support health insurance pricing and risk assessment by estimating expected medical charges for an individual based on demographic and behavioral attributes.

This project is a decision-support example for academic and technical purposes. It is not a pricing engine by itself, not a regulated actuarial system, and not a substitute for business or underwriting judgment.

## What this project demonstrates

- moving from notebooks to a `src/` production layout
- separating data loading, cleaning, validation, feature engineering, training, evaluation, and inference
- using one config file as the main source of runtime settings
- saving one deployable model artifact for consistent serving
- exposing inference through a FastAPI web API
- packaging the inference service in Docker
- logging experiments and artifacts with Weights & Biases
- keeping notebooks for exploration and `src/` for production logic

## Project architecture

The end-to-end workflow is:

1. Load raw insurance data  
2. Clean and standardize the dataframe  
3. Validate schema and domain constraints  
4. Build preprocessing features  
5. Train the regression model  
6. Evaluate performance  
7. Save the model artifact  
8. Run batch inference  
9. Serve the model through FastAPI locally or in Docker

### Main modules

- `src/load_data.py` loads raw data
- `src/clean_data.py` cleans and standardizes records
- `src/validate.py` applies schema and fail-fast validation rules
- `src/features.py` defines preprocessing for numerical and categorical columns
- `src/train.py` trains the regression model
- `src/evaluate.py` computes metrics and generates evaluation outputs
- `src/infer.py` generates predictions
- `src/utils.py` centralizes shared helpers and I/O utilities
- `src/main.py` orchestrates the full training pipeline
- `src/api.py` exposes the trained model as a FastAPI inference service

## Relevant repository structure

```text
.
├── .dockerignore
├── .gitignore
├── Dockerfile
├── LICENSE
├── README.md
├── conda-lock.yml
├── config.yaml
├── environment.yml
├── pytest.ini
├── data/
├── models/
├── notebooks/
├── reports/
├── src/
│   ├── __init__.py
│   ├── api.py
│   ├── clean_data.py
│   ├── evaluate.py
│   ├── features.py
│   ├── infer.py
│   ├── load_data.py
│   ├── main.py
│   ├── train.py
│   ├── utils.py
│   └── validate.py
└── tests/

```
## Tech stack

- Python 3.10+  
- Conda for environment management  
- pandas and NumPy for data handling  
- scikit-learn for model training and preprocessing  
- FastAPI and Pydantic for serving and request validation  
- Weights & Biases for experiment tracking and model artifact handling  
- Docker for containerized API serving  
- pytest for automated testing  

---

## Configuration

`config.yaml` is the central configuration file for the project.

It currently defines:

- paths for raw data, processed data, saved model, inference input, predictions output, and reports  
- train/test split parameters  
- schema requirements and required columns  
- target variable settings  
- domain constraints such as non-negative and positive numeric fields  
- allowed categorical values  
- feature groups for numerical and categorical preprocessing  
- logging settings  
- Weights & Biases project and artifact settings  

---

## Current configured paths

- raw data: `data/raw/insurance.csv`  
- processed data: `data/processed/clean.csv`  
- model artifact: `models/linreg_insurance.joblib`  
- inference input: `data/inference/insurance_inference.csv`  
- predictions output: `reports/predictions.csv`  

---

## Current configured target

The pipeline uses:

- `charges` as the original business target in the dataset  
- `log_charges` as the modeled target used by the pipeline  

---

## Current configured features

### Numerical:

- age  
- bmi  
- children  

### Categorical:

- sex  
- smoker  
- region  

---

Secrets should stay in `.env`, not in `config.yaml`.

---

## Example `.env`

```env
WANDB_API_KEY=your_wandb_api_key
WANDB_ENTITY=your_wandb_entity
MODEL_SOURCE=local
WANDB_MODEL_ALIAS=latest
WANDB_MODE=online
PORT=8000
```

## Setup

### 1) Create the environment

```bash
conda env create -f environment.yml
conda activate mlops_g1
```

If you want a reproducible container-style dependency install, the repo also includes conda-lock.yml.

### 2) Run the tests

```bash
pytest -q
```

### 3) Run the full pipeline

```bash
python -m src.main
```

Expected main outputs include:

cleaned data under data/processed/
trained model under models/
predictions under reports/
logs and evaluation outputs under reports/

### 4) Explore notebooks

Use the notebooks for exploration and iteration, but use src/ as the production entry point.

---

## FastAPI service

The inference service is defined in src/api.py.

### Run locally

```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
```

### Local URLs

Home: http://127.0.0.1:8000/  
Health check: http://127.0.0.1:8000/health  
Interactive docs: http://127.0.0.1:8000/docs  

---

## Prediction request format

The API expects a JSON payload with a records list.

Each record must include:

age as numeric  
bmi as numeric  
children as numeric  
sex as string  
smoker as string  
region as string  

### Example request body

```json
{
  "records": [
    {
      "age": 35,
      "sex": "male",
      "bmi": 39.71,
      "children": 4,
      "smoker": "no",
      "region": "northeast"
    }
  ]
}
```

### Example with curl

```bash
curl -X POST "http://127.0.0.1:8000/predict"   -H "Content-Type: application/json"   -d '{
    "records": [
      {
        "age": 35,
        "sex": "male",
        "bmi": 39.71,
        "children": 4,
        "smoker": "no",
        "region": "northeast"
      }
    ]
  }'
```

### Example with Python

```python
import requests

url = "http://127.0.0.1:8000/predict"
payload = {
    "records": [
        {
            "age": 35,
            "sex": "male",
            "bmi": 39.71,
            "children": 4,
            "smoker": "no",
            "region": "northeast",
        }
    ]
}

response = requests.post(url, json=payload, timeout=30)
print(response.status_code)
print(response.json())
```

---

## Expected API behavior

GET / returns a small guidance message  
GET /health returns service and model-load status  
POST /predict returns regression predictions  
the API attaches an X-Correlation-ID response header for request tracing  
when configured, inference logs can be buffered and sent to Weights & Biases  

---

## Docker

This repository includes a Dockerfile for serving the API only.

It is not designed to run the full training pipeline inside the container. The container starts uvicorn src.api:app and exposes port 8000.

### Build the image

```bash
docker build -t insurance-api:latest .
```

### Run the container with local model loading

```bash
docker run -p 8000:8000 --env-file .env --name insurance-api-container insurance-api:latest
```

### Run the container with W&B model loading

```bash
docker run -p 8000:8000 --env-file .env -e MODEL_SOURCE=wandb --name insurance-api-container insurance-api:latest
```

### Test the containerized API

http://127.0.0.1:8000/docs  
http://127.0.0.1:8000/health  

---

## Weights & Biases usage

This project uses Weights & Biases for:

experiment tracking  
metrics logging  
processed data logging  
prediction preview logging  
evaluation plot logging  
model artifact versioning  
optional inference telemetry from the API  

---

## Current W&B configuration

project: insurance-regression  
job type: training-pipeline  
model artifact name: linreg-insurance-model  

---

## Testing strategy

The tests/ folder validates the production pipeline modules.

This matters because MLOps is not only about model performance. It is also about ensuring that the full system remains stable, reproducible, and safe to change.

---

## Reproducibility rules in this repo

use config.yaml instead of hardcoding settings  
keep secrets in .env  
run the orchestrator with python -m src.main  
run tests before pushing changes  
keep one deployable serialized model artifact  
keep notebooks for exploration, not production execution  
use the API layer only for serving, validation, and observability  
