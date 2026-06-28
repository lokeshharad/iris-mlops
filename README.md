# Iris MLOps

A machine learning operations project for training and making predictions on the Iris dataset using a Random Forest classifier.

## Overview

This project demonstrates a complete ML workflow with:
- **Advanced Features**: Cross-validation, model selection, OOP architecture, comprehensive logging

## Project Structure

```
iris-mlops/
├── run_job.py           # Main entry point with CLI argument parsing
├── requirements.txt     # Python dependencies
├── .gitignore          # Git ignore patterns
├── version.txt         # Version information
├── README.md          # Documentation
├── src/
│   ├── train/          # Training package (trainer.py + __init__.py)
│   ├── predict/        # Prediction package (predictor.py + __init__.py)
│   └── reference.py    # Configuration and constants
├── data/               # Data directory for storing datasets
├── models/             # Trained model storage
└── tests/              # Unit tests directory
```

## Installation

### Prerequisites
- Python 3.7+
- pip

### Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

## Dependencies

- **numpy**: Numerical computing
- **pandas**: Data manipulation and analysis
- **scikit-learn**: Machine learning library

## Usage

### Training the Model

Train a new model (full pipeline: CV, model selection, final training):

```bash
python run_job.py -op train
```

What this does:
- Loads dataset from `data/iris.data` (see `src/reference.py` for paths)
- Preprocesses data and runs 5-fold stratified CV on training folds
- Performs model selection across 5 candidate models (LR, DT, RF, SVC, KNN)
- Trains the selected model on the full training set and evaluates the test set
- Serializes the final model to `models/model.pkl`

### Making Predictions (CLI via `run_job.py`)

The project exposes prediction modes through `run_job.py` to keep a single entrypoint.

Single-sample prediction (provide four numeric features):

```bash
python run_job.py -op single_predict --features 5.1 3.5 1.4 0.2
```

Batch prediction from a CSV file (first 4 columns are features; optional 5th column is `target` for evaluation):

```bash
python run_job.py -op batch_predict --csv data/iris_batch_test.csv
```

Notes:
- `single_predict` requires `--features` with exactly 4 numeric values.
- `batch_predict` requires `--csv` pointing to a CSV file. If the CSV contains a 5th column, the pipeline will compute evaluation metrics.

You can also call the `src.predict` package directly for more advanced usage (see `src/predict/__init__.py`).

## Model Architecture

### Training Pipeline (train.py)

OOP-based `Trainer` class with:
- Data loading and preprocessing
- Stratified train-test split (80-20)
- Model selection with cross-validation
- Final model training and evaluation
- Model serialization with joblib

### Prediction Pipeline (predict.py)

OOP-based `Predictor` class with:
- Single sample predictions
- Batch predictions from CSV
- Evaluation with metrics and classification reports
- Probability predictions (when available)
- Comprehensive error handling and logging

### Model Versioning

- Trained models are saved with timestamped filenames in the `models/` folder, e.g. `model_20260623T123456Z.pkl`.
- A `latest.txt` file in `models/` points to the most recent model file.
- The training pipeline keeps only the most recent 5 model files; older models are automatically removed to limit storage.
- The prediction pipeline (`Predictor.load_model`) will automatically load the latest available model if an explicit model file is not provided.

## Data Files

- `data/iris.csv` - Original iris dataset
- `data/iris_batch_test.csv` - Sample batch for testing predictions

## Version

Current version: **1.3.0.1**

See `version.txt` for detailed version information and component descriptions.

## Data Format

### Iris Dataset Columns
1. **sepal_length**: Sepal length in cm
2. **sepal_width**: Sepal width in cm
3. **petal_length**: Petal length in cm
4. **petal_width**: Petal width in cm
5. **target**: Iris species (0, 1, or 2)

### Expected CSV Format
```
5.1,3.5,1.4,0.2,0
7.0,3.2,4.7,1.4,1
6.3,3.3,6.0,2.5,2
```

## Model Details

- **Algorithm**: Random Forest Classifier
- **Number of trees**: 100
- **Random state**: 42 (for reproducibility)
- **Features**: 4 (sepal_length, sepal_width, petal_length, petal_width)
- **Classes**: 3 (Iris setosa, Iris versicolor, Iris virginica)

## How It Works

### Training Pipeline
```
# Iris MLOps

This repository contains an ML pipeline for the Iris dataset with a focus on reproducible training, model selection, versioned model artifacts, and simple CLI-driven prediction workflows.

Repository snapshot
-------------------
- `run_job.py` — single CLI entrypoint that dispatches operations (`train`, `single_predict`, `batch_predict`).
- `src/train/` — package exposing `Trainer` and `main()` for training (contains `trainer.py`).
- `src/predict/` — package exposing `Predictor` and `main()` for prediction (contains `predictor.py`).
- `src/reference.py` — small constants for `DATA_FOLDER`, `MODEL_FOLDER`, `DATA_FILE`, `MODEL_FILE`.
- `data/` — data files. `iris_batch_test.csv` included as an example batch with target column for evaluation.
- `models/` — serialized models (created by training). Training writes timestamped model files and a `latest.txt` pointer.
- `requirements.txt` — Python package dependencies.
- `version.txt` — project version and release notes.

Quick start
-----------
1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Train (runs full pipeline: CV, model selection, final training, serialization):

```bash
python run_job.py -op train
```

3. Single prediction (provide four features):

```bash
python run_job.py -op single_predict --features 5.1 3.5 1.4 0.2
```

4. Batch prediction from CSV (first 4 columns are features; optional 5th column `target` for evaluation):

```bash
python run_job.py -op batch_predict --csv data/iris_batch_test.csv
```

Data format
-----------
The CSV files expected by the pipeline should have at least the following columns (no header required for training data if using the default `data/iris.data`):

- sepal_length, sepal_width, petal_length, petal_width
- Optional fifth column: `target` (0, 1, 2) — if present, batch prediction will also compute evaluation metrics.

Model versioning
----------------
- Trained models are saved to the `models/` directory with timestamped filenames: `model_YYYYMMDDTHHMMSSZ.pkl`.
- A `latest.txt` file is written to `models/` and contains the filename of the most recent model.
- The training pipeline retains only the latest 5 model files; older files are removed automatically.
- The prediction pipeline loads the model as follows:
	1. If `models/model.pkl` (configured `MODEL_FILE`) exists and is passed, it is loaded.
	2. Else, if `models/latest.txt` exists, its referenced file is loaded.
	3. Else, the most recent `*.pkl` file in `models/` is selected.

CLI reference (`run_job.py`)
--------------------------
- `-op train` — Run training pipeline (no extra args).
- `-op single_predict --features a b c d` — Single prediction with four numeric features.
- `-op batch_predict --csv path/to/file.csv` — Batch predictions from CSV; optional `target` column triggers evaluation.

Design notes
------------
- The training pipeline runs cross-validation using `StratifiedKFold` and uses `cross_val_predict` for CV predictions during model selection.
- Candidate models evaluated: LogisticRegression (with StandardScaler), DecisionTreeClassifier, RandomForestClassifier, SVC (with StandardScaler), KNeighborsClassifier (with StandardScaler).
- Model selection picks the best model by weighted F1 score.
- Logging is used extensively throughout the code for visibility into each pipeline stage.

Files to check when debugging
---------------------------
- `models/latest.txt` — indicates the latest model file used for prediction.
- `data/iris_batch_test.csv` — example CSV with target column for evaluation.
- `version.txt` — project version and release notes.

Version
-------
Current version: **1.3.0.1** — see `version.txt` for recent change notes (MLflow integration for training and prediction).

Troubleshooting
---------------
- If training fails due to missing data: ensure `data/iris.data` or a suitable CSV exists in `data/`.
- If prediction fails complaining about missing models: run training first to create model artifacts.
- For single prediction, ensure four numeric features are provided.

Running tests
-------------
Run unit tests (if present):

```bash
pytest tests/
```

Contact / Author
----------------
Add author/contact information here.
