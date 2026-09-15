# Incremental ML Pipeline on Databricks

A small production-oriented machine learning pipeline built with:

- Databricks
- Databricks Declarative Automation Bundles
- MLflow
- Delta Lake
- Python
- scikit-learn
- GitHub
- GitHub Actions

## Architecture

Fake Data
    ↓
Feature Engineering
    ↓
MLflow Model
    ↓
Incremental Predictions
    ↓
Delta Prediction Table

## Pipeline

The pipeline starts with 50 synthetic customer records.

Every subsequent run generates 2 new records.

Only previously unseen customers are sent to the ML model for prediction.

## Data

The synthetic dataset contains:

- customer_id
- age
- income
- previous_purchases
- avg_purchase
- will_buy

## Feature Engineering

One simple feature is created:

total_spending = previous_purchases * avg_purchase

## Machine Learning

A RandomForestClassifier is trained using scikit-learn.

The model is tracked and registered using MLflow.

## Incremental Inference

The prediction pipeline checks which customer IDs have already been predicted.

Only new customers are processed.

Predictions are appended to a dedicated Delta table.

## CI/CD

GitHub Actions runs:

1. Unit tests
2. Databricks Bundle validation

Production deployment can be added after validation.

## Project Structure

```text
incremental-ml-databricks/
│
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── resources/
│   └── job.yml
│
├── src/
│   ├── generate_data.py
│   ├── feature_engineering.py
│   ├── train.py
│   └── predict.py
│
├── tests/
│   └── test_features.py
│
├── databricks.yml
├── requirements.txt
├── .gitignore
└── README.md