import numpy as np
import pandas as pd
import joblib
import warnings
import logging
import os
from src.reference import MODEL_FOLDER, MODEL_FILE

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

            # Build list of candidate .pkl files to try (latest pointer first if present)
            candidates = []
            if candidate:
                candidates.append(candidate)

            pkls = [os.path.join(model_folder, f) for f in os.listdir(model_folder) if f.endswith('.pkl')]
            # sort by modification time descending
            pkls.sort(key=lambda p: os.path.getmtime(p), reverse=True)
            for p in pkls:
                if p not in candidates:
                    candidates.append(p)

            if not candidates:
                raise FileNotFoundError(f"No model files found in {model_folder}")

            last_err = None
            for cand in candidates:
                logging.info(f"Attempting to load model file: {cand}")
                try:
                    self.model = joblib.load(cand)
                    logging.info(f"Model loaded successfully. Model type: {type(self.model).__name__}")
                    return self.model
                except Exception as e:
                    logging.warning(f"Failed to load model {cand}: {e}. Trying next candidate if available.")
                    last_err = e

            # If we get here, none of the candidates loaded successfully
            logging.error(f"All candidate model files failed to load in {model_folder}")
            if last_err:
                raise last_err
            else:
                raise FileNotFoundError(f"No model files found in {model_folder}")
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
