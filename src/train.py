import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import logging
from .reference import DATA_FOLDER, MODEL_FOLDER, DATA_FILE, MODEL_FILE

logging.basicConfig(level=logging.INFO)

def read_data(file_path):
    data = pd.read_csv(file_path)
    data.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'target']
    logging.info(f"Data read from: {file_path}")
    return data

def preprocess_data(data):
    # Example preprocessing steps - modify as needed
    data = data.dropna()
    X = data.drop('target', axis=1)
    y = data['target']
    logging.info(f"Data preprocessed. Shape: {X.shape}")
    return X, y

def train_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    logging.info("Model trained.")
    return model

def serialize_model(model, file_path):
    import joblib
    joblib.dump(model, file_path)
    logging.info(f"Model serialized to: {file_path}")

def main():
    data = read_data(f"{DATA_FOLDER}/{DATA_FILE}")
    X, y = preprocess_data(data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = train_model(X_train, y_train)
    serialize_model(model, f"{MODEL_FOLDER}/{MODEL_FILE}")