import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import logging
import os
import joblib
from datetime import datetime

from src.reference import DATA_FOLDER, MODEL_FOLDER, DATA_FILE, MODEL_FILE

logging.basicConfig(level=logging.INFO)


class Trainer:
    """OOP-style trainer that encapsulates data loading, training, CV and evaluation."""

    def __init__(self, data_path, model_path, estimator=None, cv_splits=5):
        self.data_path = data_path
        self.model_path = model_path
        self.estimator = estimator or RandomForestClassifier(n_estimators=100, random_state=42)
        self.cv_splits = cv_splits
        self.model = None
        logging.info(f"Trainer initialized: data_path={data_path}, model_path={model_path}, cv_splits={cv_splits}")

    def read_data(self):
        logging.info(f"Reading data from: {self.data_path}")
        try:
            data = pd.read_csv(self.data_path, header=None)
            data.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'target']
            logging.info(f"Data loaded successfully. Shape: {data.shape}")
            return data
        except Exception as e:
            logging.error(f"Failed to read data: {e}")
            raise

    def preprocess(self, data):
        logging.info(f"Starting data preprocessing. Initial shape: {data.shape}")
        data = data.dropna()
        logging.info(f"After removing NaNs: {data.shape}")
        X = data.drop('target', axis=1)
        y = data['target']
        logging.info(f"Data preprocessed. Feature matrix shape: {X.shape}, target shape: {y.shape}")
        logging.info(f"Target classes: {y.unique()}")
        return X, y

    def evaluate_cv(self, X, y):
        cv = StratifiedKFold(n_splits=self.cv_splits, shuffle=True, random_state=42)
        y_pred = cross_val_predict(self.estimator, X, y, cv=cv)

        acc = accuracy_score(y, y_pred)
        prec = precision_score(y, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y, y_pred, average='weighted', zero_division=0)

        logging.info(f"Cross-validation results (n_splits={self.cv_splits}): accuracy={acc:.4f}, precision={prec:.4f}, recall={rec:.4f}, f1={f1:.4f}")
        logging.info("Classification report (CV):\n" + classification_report(y, y_pred, zero_division=0))
        return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}

    def model_selection(self, X, y, metric='f1'):
        """Evaluate a set of candidate models using CV and pick the best by `metric` (weighted)."""
        logging.info(f"Starting model selection with {self.cv_splits}-fold CV on {len(y)} samples.")
        candidates = {
            'logistic_regression': Pipeline([('scaler', StandardScaler()), ('lr', LogisticRegression(max_iter=1000))]),
            'decision_tree': DecisionTreeClassifier(random_state=42),
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'svc': Pipeline([('scaler', StandardScaler()), ('svc', SVC())]),
            'knn': Pipeline([('scaler', StandardScaler()), ('knn', KNeighborsClassifier())])
        }
        logging.info(f"Candidate models: {list(candidates.keys())}")

        results = {}
        cv = StratifiedKFold(n_splits=self.cv_splits, shuffle=True, random_state=42)

        for name, estimator in candidates.items():
            logging.info(f"Evaluating model: {name}")
            try:
                y_pred = cross_val_predict(estimator, X, y, cv=cv)
                acc = accuracy_score(y, y_pred)
                prec = precision_score(y, y_pred, average='weighted', zero_division=0)
                rec = recall_score(y, y_pred, average='weighted', zero_division=0)
                f1 = f1_score(y, y_pred, average='weighted', zero_division=0)
                results[name] = {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1}
                logging.info(f"Model {name} CV -> accuracy={acc:.4f}, precision={prec:.4f}, recall={rec:.4f}, f1={f1:.4f}")
                logging.info(f"Classification report ({name} CV):\n" + classification_report(y, y_pred, zero_division=0))
            except Exception as e:
                logging.error(f"Error evaluating model {name}: {e}")
                raise

        # pick best by requested metric
        best_name = max(results.keys(), key=lambda n: results[n].get(metric, 0))
        best_metric_value = results[best_name].get(metric)
        logging.info(f"\n=== MODEL SELECTION SUMMARY ===")
        for name, metrics in sorted(results.items(), key=lambda x: x[1].get(metric, 0), reverse=True):
            logging.info(f"{name}: {metric}={metrics.get(metric):.4f}")
        logging.info(f"\nSelected best model: '{best_name}' with {metric}={best_metric_value:.4f}")
        logging.info(f"=== END SELECTION ===")

        # return a fresh estimator instance for the best model
        best_estimator = candidates[best_name]
        return best_name, best_estimator, results

    def train(self, X_train, y_train):
        logging.info(f"Starting training on {len(y_train)} training samples with {X_train.shape[1]} features.")
        try:
            self.model = self.estimator
            self.model.fit(X_train, y_train)
            logging.info(f"Model trained successfully.")
            logging.info(f"Model type: {type(self.model).__name__}")
            return self.model
        except Exception as e:
            logging.error(f"Error during model training: {e}")
            raise

    def evaluate_test(self, y_true, y_pred):
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        logging.info(f"Test results: accuracy={acc:.4f}, precision={prec:.4f}, recall={rec:.4f}, f1={f1:.4f}")
        logging.info("Classification report (Test):\n" + classification_report(y_true, y_pred, zero_division=0))
        return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}

    @staticmethod
    def _ensure_folder_exists(folder_path):
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            logging.info(f"Folder created: {folder_path}")
        else:
            logging.info(f"Folder already exists: {folder_path}")

    def serialize_model(self):
        folder_path = os.path.dirname(self.model_path)
        self._ensure_folder_exists(folder_path)

        # Create timestamped model filename for versioning
        timestamp = datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')
        model_filename = f"model_{timestamp}.pkl"
        model_fullpath = os.path.join(folder_path, model_filename)

        logging.info(f"Serializing model to: {model_fullpath}")
        try:
            joblib.dump(self.model, model_fullpath)

            # Update latest pointer file
            latest_file = os.path.join(folder_path, 'latest.txt')
            try:
                with open(latest_file, 'w') as f:
                    f.write(model_filename)
            except Exception:
                logging.warning(f"Could not write latest pointer file: {latest_file}")

            logging.info(f"Model successfully serialized to: {model_fullpath}")

            # Keep only the most recent N models (keep_latest)
            keep_latest = 5
            all_models = [f for f in os.listdir(folder_path) if f.endswith('.pkl')]
            all_models_full = [os.path.join(folder_path, f) for f in all_models]
            # Sort by modification time descending
            all_models_full.sort(key=lambda p: os.path.getmtime(p), reverse=True)
            # Remove older models beyond keep_latest
            for old_model in all_models_full[keep_latest:]:
                try:
                    os.remove(old_model)
                    logging.info(f"Removed old model: {old_model}")
                except Exception as e:
                    logging.warning(f"Failed to remove old model {old_model}: {e}")

        except Exception as e:
            logging.error(f"Failed to serialize model: {e}")
            raise

    def run(self):
        logging.info("\n" + "=" * 60)
        logging.info("TRAINING PIPELINE STARTED")
        logging.info("=" * 60)
        
        try:
            # Load and preprocess data
            logging.info("\n[STAGE 1] Loading and preprocessing data...")
            data = self.read_data()
            X, y = self.preprocess(data)
            
            # Train-test split
            logging.info("\n[STAGE 2] Splitting data into train/test sets (80/20)...")
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
            logging.info(f"Train set: {X_train.shape[0]} samples, Test set: {X_test.shape[0]} samples")
            
            # Model selection on training data
            logging.info("\n[STAGE 3] Performing model selection...")
            best_name, best_estimator, selection_results = self.model_selection(X_train, y_train, metric='f1')
            self.estimator = best_estimator
            
            # Train final model and evaluate on test set
            logging.info("\n[STAGE 4] Training final model...")
            self.train(X_train, y_train)
            
            logging.info("\n[STAGE 5] Evaluating on test set...")
            y_test_pred = self.model.predict(X_test)
            self.evaluate_test(y_test, y_test_pred)
            
            logging.info("\n[STAGE 6] Serializing trained model...")
            self.serialize_model()
            
            logging.info("\n" + "=" * 60)
            logging.info("TRAINING PIPELINE COMPLETED SUCCESSFULLY")
            logging.info("=" * 60 + "\n")
        except Exception as e:
            logging.error(f"\nTraining pipeline failed with error: {e}")
            logging.error("=" * 60)
            raise
