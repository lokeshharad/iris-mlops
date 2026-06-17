# Iris MLOps

A machine learning operations project for training and making predictions on the Iris dataset using a Random Forest classifier.

## Overview

This project demonstrates a complete ML workflow with:
- **Training**: Build and train a Random Forest model on the Iris dataset
- **Prediction**: Make predictions on new data using the trained model
- **Command-line interface**: Easy-to-use CLI for running training and prediction jobs

## Project Structure

```
iris-mlops/
├── run_job.py           # Main entry point with CLI argument parsing
├── requirements.txt     # Python dependencies
├── .gitignore          # Git ignore patterns
├── src/
│   ├── train.py        # Model training logic
│   ├── predict.py      # Prediction logic
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

Train a new Random Forest model on the Iris dataset:

```bash
python run_job.py --operation train
```

or

```bash
python run_job.py -op train
```

This will:
1. Read the Iris dataset from `data/iris.data`
2. Preprocess the data (handle missing values)
3. Train a Random Forest classifier with 100 estimators
4. Save the model to `models/model.pkl`

### Making Predictions

Make predictions on new data:

```bash
python run_job.py --operation predict --input_array "[5.1, 3.5, 1.4, 0.2]"
```

or

```bash
python run_job.py -op predict -inp "[5.1, 3.5, 1.4, 0.2]"
```

**Parameters:**
- `--operation` / `-op`: `train` or `predict` (required)
- `--input_array` / `-inp`: Input features as a string representation of a list for prediction (optional for train, required for predict)

**Example input formats:**
- Single sample: `"[5.1, 3.5, 1.4, 0.2]"`
- Multiple samples: Pass array values matching Iris feature dimensions (sepal_length, sepal_width, petal_length, petal_width)

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
raw_data → read_data() → preprocess_data() → train_model() → serialize_model()
```

1. **Read Data**: Load CSV file and assign column names
2. **Preprocess**: Drop missing values, split features (X) and target (y)
3. **Train**: Fit Random Forest model
4. **Serialize**: Save trained model using joblib

### Prediction Pipeline
```
input_array → load_model() → predict() → output
```

1. **Load Model**: Retrieve trained model from disk
2. **Reshape Input**: Convert input array to proper format
3. **Predict**: Generate predictions using the model

## Configuration

Configuration constants are defined in `src/reference.py`:

```python
BASE_FOLDER = "iris-mlops"
DATA_FOLDER = f"{BASE_FOLDER}/data"
MODEL_FOLDER = f"{BASE_FOLDER}/models"
DATA_FILE = "iris.data"
MODEL_FILE = "model.pkl"
```

## Logging

All operations are logged with INFO level:
- Data loading and preprocessing steps
- Model training progress
- Predictions made

## Testing

Run tests from the project root:

```bash
pytest tests/
```

## Project Workflow

1. **Prepare data**: Place your Iris dataset in `data/` directory
2. **Train model**: Run `python run_job.py -op train`
3. **Make predictions**: Run `python run_job.py -op predict -inp "[5.1, 3.5, 1.4, 0.2]"`
4. **Evaluate results**: Check logged outputs and predictions

## Troubleshooting

### Model File Not Found
- Ensure `models/` directory exists
- Train the model first with `--operation train`

### Input Array Error
- Verify input array format: `"[float1, float2, float3, float4]"`
- Ensure 4 features are provided (matching Iris dataset)

### Data File Not Found
- Place `iris.data` in the `data/` directory
- Verify CSV format matches expected structure

## Future Enhancements

- [ ] Model hyperparameter tuning
- [ ] Cross-validation metrics
- [ ] Multiple model comparison
- [ ] RESTful API for predictions
- [ ] Docker containerization
- [ ] CI/CD pipeline integration
- [ ] Model versioning and tracking

## License

[Add your license here]

## Author

[Add author information]
