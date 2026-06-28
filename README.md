# Iris MLOps

Iris MLOps is a small end-to-end machine learning project for training and predicting the Iris dataset with a model selection pipeline and MLflow-based model registry tracking.

## Overview

This project demonstrates:
- model selection across several classifiers
- cross-validation and evaluation metrics
- MLflow experiment tracking
- MLflow Model Registry versioning for deployment-ready models
- a simple CLI for training and prediction

## Project Structure

```text
iris-mlops/
├── run_job.py              # CLI entry point for train/predict operations
├── requirements.txt       # Python dependencies
├── version.txt            # Project version and release history
├── README.md              # Project documentation
├── data/                  # Input datasets
├── src/
│   ├── train/             # Training pipeline and MLflow registration logic
│   ├── predict/           # Prediction pipeline using registry-backed models
│   └── reference.py       # Shared configuration constants
└── tests/                 # Test directory
```

## Installation

### Prerequisites
- Python 3.9+
- pip

### Setup

```bash
pip install -r requirements.txt
```

## Usage

### Train a model

```bash
python run_job.py -op train
```

This workflow will:
- load the Iris dataset from data/iris.data
- preprocess the data
- evaluate multiple candidate models with cross-validation
- select the best model by weighted F1 score
- log metrics and parameters to MLflow
- register the selected model in the MLflow Model Registry

### Single prediction

```bash
python run_job.py -op single_predict --features 5.1 3.5 1.4 0.2
```

### Batch prediction

```bash
python run_job.py -op batch_predict --csv data/iris_batch_test.csv
```

## MLflow Workflow

The current implementation uses the MLflow Model Registry as the source of truth for inference.

### Start MLflow UI

```bash
mlflow ui
```

Then open:

```text
http://127.0.0.1:5000
```

### What happens during training
- an MLflow experiment is created
- training metrics are logged
- the selected model is registered under a dynamic model name based on the winning model and its metric value
- the model version is promoted to the Production stage

### What happens during prediction
- prediction loads the latest Production model from the MLflow registry
- no local model pickle folder is required for inference

## Data Format

The default training data uses the following columns:
1. sepal_length
2. sepal_width
3. petal_length
4. petal_width
5. target

Batch prediction files may include the same 5 columns, where the fifth column is optional target data for evaluation.

## Model Selection

The training pipeline evaluates the following candidate classifiers:
- Logistic Regression
- Decision Tree
- Random Forest
- SVM
- K-Nearest Neighbors

The best model is selected using weighted F1 score.

## Version

Current version: 1.4.0.0

See version.txt for detailed release notes.
