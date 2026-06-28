import numpy as np
import warnings
import logging
import mlflow
import mlflow.sklearn
from src.reference import MLFLOW_MODEL_STAGE

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)


class Predictor:
    """OOP-style predictor for making predictions with trained models."""

    def __init__(self):
        self.model = None
        logging.info("Predictor initialized")

    def load_model(self):
        """Load the latest Production model from the MLflow registry."""
        logging.info("Loading model from MLflow registry")
        try:
            client = mlflow.tracking.MlflowClient()
            versions = client.search_model_versions("name LIKE 'iris-%'")
            if not versions:
                raise FileNotFoundError("No registered MLflow models found")

            latest_version = max(versions, key=lambda m: int(m.version))
            registry_uri = f"models:/{latest_version.name}/{MLFLOW_MODEL_STAGE}"
            self.model = mlflow.sklearn.load_model(registry_uri)
            logging.info(f"Model loaded successfully from MLflow registry. Model type: {type(self.model).__name__}")
            return self.model
        except Exception as e:
            logging.error(f"Failed to load model from MLflow registry: {e}")
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
