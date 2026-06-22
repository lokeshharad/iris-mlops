import numpy as np
import pandas as pd
import joblib
import warnings
import logging
import os
from .reference import MODEL_FOLDER, MODEL_FILE

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)


class Predictor:
    """OOP-style predictor for making predictions with trained models."""

    def __init__(self, model_path):
        self.model_path = model_path
        self.model = None
        logging.info(f"Predictor initialized: model_path={model_path}")

    def load_model(self):
        """Load trained model from disk."""
        logging.info(f"Loading model from: {self.model_path}")
        # If the provided model path exists, load it.
        try:
            # If a specific model path exists, try loading it first.
            if os.path.exists(self.model_path):
                try:
                    self.model = joblib.load(self.model_path)
                    logging.info(f"Model loaded successfully from path. Model type: {type(self.model).__name__}")
                    return self.model
                except Exception as e:
                    logging.warning(f"Failed loading configured model {self.model_path}: {e}. Will try latest model in folder.")

            # Otherwise try to read latest pointer in model folder
            model_folder = os.path.dirname(self.model_path)
            latest_file = os.path.join(model_folder, 'latest.txt')
            candidate = None
            if os.path.exists(latest_file):
                try:
                    with open(latest_file, 'r') as f:
                        fname = f.read().strip()
                        candidate = os.path.join(model_folder, fname)
                        if not os.path.exists(candidate):
                            candidate = None
                except Exception:
                    candidate = None

            # If no latest pointer or candidate missing, pick most recent .pkl in folder
            if candidate is None:
                pkls = [os.path.join(model_folder, f) for f in os.listdir(model_folder) if f.endswith('.pkl')]
                if not pkls:
                    raise FileNotFoundError(f"No model files found in {model_folder}")
                pkls.sort(key=lambda p: os.path.getmtime(p), reverse=True)
                candidate = pkls[0]

            logging.info(f"Loading most recent model file: {candidate}")
            self.model = joblib.load(candidate)
            logging.info(f"Model loaded successfully. Model type: {type(self.model).__name__}")
            return self.model
        except Exception as e:
            logging.error(f"Failed to load model: {e}")
            raise

    def predict_single(self, input_array):
        """Make prediction for a single sample."""
        if self.model is None:
            logging.error("Model not loaded. Call load_model() first.")
            raise ValueError("Model not loaded.")
        
        logging.info(f"Making prediction for single sample: {input_array}")
        try:
            input_data = np.array(input_array).reshape(1, -1)
            prediction = self.model.predict(input_data)
            logging.info(f"Prediction: class={prediction[0]}")
            
            # Try to get probability predictions if available
            if hasattr(self.model, 'predict_proba'):
                proba = self.model.predict_proba(input_data)
                logging.info(f"Prediction probabilities: {proba[0]}")
                return {"prediction": prediction[0], "probabilities": proba[0]}
            else:
                logging.debug("Model does not support probability predictions.")
                return {"prediction": prediction[0]}
        except Exception as e:
            logging.error(f"Error during prediction: {e}")
            raise

    def predict_batch(self, input_data):
        """Make predictions for a batch of samples."""
        if self.model is None:
            logging.error("Model not loaded. Call load_model() first.")
            raise ValueError("Model not loaded.")
        
        logging.info(f"Making batch predictions for {len(input_data)} samples.")
        try:
            predictions = self.model.predict(input_data)
            logging.info(f"Batch prediction completed. Unique classes: {np.unique(predictions)}")
            logging.info(f"Prediction distribution: {dict(zip(*np.unique(predictions, return_counts=True)))}")
            
            # Try to get probability predictions if available
            if hasattr(self.model, 'predict_proba'):
                probabilities = self.model.predict_proba(input_data)
                logging.info(f"Probability predictions available for {len(probabilities)} samples.")
                return {"predictions": predictions, "probabilities": probabilities}
            else:
                logging.debug("Model does not support probability predictions.")
                return {"predictions": predictions}
        except Exception as e:
            logging.error(f"Error during batch prediction: {e}")
            raise

    def evaluate(self, X_test, y_test):
        """Evaluate predictions on a test set with metrics and classification report."""
        if self.model is None:
            logging.error("Model not loaded. Call load_model() first.")
            raise ValueError("Model not loaded.")
        
        logging.info(f"\\nEvaluating on test set ({len(y_test)} samples)...")
        try:
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
            
            y_pred = self.model.predict(X_test)
            # Harmonize label types if necessary (strings vs numeric)
            y_test_arr = np.asarray(y_test)
            y_pred_arr = np.asarray(y_pred)

            def _is_numeric(arr):
                try:
                    return np.issubdtype(arr.dtype, np.number)
                except Exception:
                    return False

            y_test_eval = y_test_arr
            y_pred_eval = y_pred_arr

            if _is_numeric(y_test_arr) and not _is_numeric(y_pred_arr):
                # Ground truth numeric, predictions string -> map numeric -> class names using model.classes_
                if hasattr(self.model, 'classes_'):
                    classes = np.asarray(self.model.classes_)
                    try:
                        y_test_eval = np.array([classes[int(x)] for x in y_test_arr])
                        logging.info("Converted numeric y_test to string class labels to match predictions.")
                    except Exception as e:
                        logging.error(f"Failed to map numeric y_test to class labels: {e}")
                        raise
            elif not _is_numeric(y_test_arr) and _is_numeric(y_pred_arr):
                # Ground truth string, predictions numeric -> map predictions -> class names using model.classes_
                if hasattr(self.model, 'classes_'):
                    classes = np.asarray(self.model.classes_)
                    try:
                        y_pred_eval = np.array([classes[int(x)] for x in y_pred_arr])
                        logging.info("Converted numeric predictions to string class labels to match ground truth.")
                    except Exception as e:
                        logging.error(f"Failed to map numeric predictions to class labels: {e}")
                        raise

            acc = accuracy_score(y_test_eval, y_pred_eval)
            prec = precision_score(y_test_eval, y_pred_eval, average='weighted', zero_division=0)
            rec = recall_score(y_test_eval, y_pred_eval, average='weighted', zero_division=0)
            f1 = f1_score(y_test_eval, y_pred_eval, average='weighted', zero_division=0)
            
            logging.info(f"Test Metrics: accuracy={acc:.4f}, precision={prec:.4f}, recall={rec:.4f}, f1={f1:.4f}")
            logging.info(f"Classification report (Test):\\n" + classification_report(y_test_eval, y_pred_eval, zero_division=0))

            return {
                "accuracy": acc,
                "precision": prec,
                "recall": rec,
                "f1": f1,
                "predictions": y_pred,
                "report": classification_report(y_test_eval, y_pred_eval, zero_division=0, output_dict=True)
            }
        except Exception as e:
            logging.error(f"Error during evaluation: {e}")
            raise

    def run_single_prediction(self, input_array):
        """Convenient wrapper: load model and predict for single sample."""
        logging.info("\\n" + "="*60)
        logging.info("PREDICTION PIPELINE - SINGLE SAMPLE")
        logging.info("="*60)
        
        try:
            self.load_model()
            result = self.predict_single(input_array)
            logging.info("="*60 + "\\n")
            return result
        except Exception as e:
            logging.error(f"Prediction pipeline failed: {e}")
            raise

    def run_batch_prediction(self, input_data):
        """Convenient wrapper: load model and predict for batch samples."""
        logging.info("\\n" + "="*60)
        logging.info("PREDICTION PIPELINE - BATCH SAMPLES")
        logging.info("="*60)
        
        try:
            self.load_model()
            result = self.predict_batch(input_data)
            logging.info("="*60 + "\\n")
            return result
        except Exception as e:
            logging.error(f"Prediction pipeline failed: {e}")
            raise


def main(predict_type, input_path=None, input_array=None):
    """Main entry point for predictions.
    
    Args:
        predict_type: 'single_predict' for single sample or 'batch_predict' for batch from CSV
        input_path: Path to CSV file (for batch_predict)
        input_array: Array of features (for single_predict)
    """
    logging.info("\n" + "#"*60)
    logging.info("# IRIS MLOPS - PREDICTION PIPELINE")
    logging.info("#"*60)
    logging.info(f"Mode: {predict_type}")
    
    try:
        model_path = os.path.join(MODEL_FOLDER, MODEL_FILE)
        predictor = Predictor(model_path=model_path)
        
        if predict_type == 'batch_predict':
            if input_path is None:
                logging.error("CSV path required for batch_predict")
                raise ValueError("CSV path not provided")
            
            logging.info(f"Loading batch data from: {input_path}")

            # Detect whether CSV has a header row. Training data used headerless CSVs,
            # so prefer reading with header=None when first row looks numeric.
            try:
                sample = pd.read_csv(input_path, nrows=1, header=None)
                first_row = sample.iloc[0].tolist()

                def _is_number(x):
                    try:
                        float(x)
                        return True
                    except Exception:
                        return False

                if all(_is_number(v) for v in first_row):
                    df = pd.read_csv(input_path, header=None)
                    logging.info("Detected headerless CSV; reading with header=None")
                else:
                    df = pd.read_csv(input_path, header=0)
                    logging.info("Detected CSV with header row; reading with header=0")
            except Exception:
                logging.warning("Failed to auto-detect CSV header. Reading with header=None")
                df = pd.read_csv(input_path, header=None)
            
            # Extract features (first 4 columns)
            feature_cols = df.columns[:4]
            X = df[feature_cols].values
            logging.info(f"Loaded {len(X)} samples with {X.shape[1]} features")
            
            # Check if target column exists (5th column)
            y = None
            if len(df.columns) > 4:
                y = df[df.columns[4]].values
                logging.info("Target column found for evaluation")
            
            result = predictor.run_batch_prediction(X)
            
            # Evaluate if target available
            if y is not None:
                logging.info("\nEvaluating predictions...")
                predictor.load_model()
                eval_result = predictor.evaluate(X, y)
                logging.info(f"Evaluation Metrics:")
                logging.info(f"  Accuracy:  {eval_result['accuracy']:.4f}")
                logging.info(f"  Precision: {eval_result['precision']:.4f}")
                logging.info(f"  Recall:    {eval_result['recall']:.4f}")
                logging.info(f"  F1 Score:  {eval_result['f1']:.4f}")
                result['metrics'] = eval_result
        
        elif predict_type == 'single_predict':
            if input_array is None:
                input_array = [5.1, 3.5, 1.4, 0.2]  # Default iris sample
                logging.info(f"Using default input sample: {input_array}")
            
            result = predictor.run_single_prediction(input_array)
            logging.info(f"Result: {result}")
        
        else:
            logging.error(f"Unknown predict type: {predict_type}")
            raise ValueError(f"Invalid predict_type: {predict_type}")
        
        logging.info("\n# PIPELINE EXECUTION COMPLETE")
        logging.info("#"*60 + "\n")
        return result
    except Exception as e:
        logging.error(f"\n# PIPELINE EXECUTION FAILED: {e}")
        logging.error("#"*60 + "\n")
        raise


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("\nUsage:")
        print("  python -m src.predict single_predict [feature1 feature2 feature3 feature4]")
        print("  python -m src.predict batch_predict <csv_file_path>")
        print("\nExamples:")
        print("  python -m src.predict single_predict 5.1 3.5 1.4 0.2")
        print("  python -m src.predict batch_predict data/iris_test.csv")
        sys.exit(1)
    
    predict_type = sys.argv[1]
    
    if predict_type == 'single_predict':
        input_array = None
        if len(sys.argv) > 2:
            try:
                input_array = [float(x) for x in sys.argv[2:6]]
            except ValueError:
                logging.error("Invalid numeric input for single_predict")
                sys.exit(1)
        main(predict_type, input_array=input_array)
    
    elif predict_type == 'batch_predict':
        if len(sys.argv) < 3:
            logging.error("CSV file path required for batch_predict")
            print("Usage: python -m src.predict batch_predict <csv_file_path>")
            sys.exit(1)
        csv_path = sys.argv[2]
        main(predict_type, input_path=csv_path)
    
    else:
        logging.error(f"Unknown predict type: {predict_type}")
        print(f"Error: Unknown predict type '{predict_type}'")
        sys.exit(1)
