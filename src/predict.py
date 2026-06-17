import numpy as np
import pandas as pd
import joblib
import warnings
warnings.filterwarnings("ignore")
import logging  

logging.basicConfig(level=logging.INFO)


def load_model(file_path):
    model = joblib.load(file_path)
    logging.info(f"Model loaded from: {file_path}")
    return model

def predict(model, input_data):
    logging.info(f"Making predictions for input data: {input_data}")
    predictions = model.predict(input_data)
    return predictions

def main(input_array):
    # Load the trained model
    model = load_model("project/models/model.pkl")
    
    # Example input data for prediction (replace with actual input)
    input_data = np.array(input_array).reshape(1, -1)  # Reshape for a single sample
    # Make predictions
    predictions = predict(model, input_data)
    
    logging.info(f"Predicted class: {predictions[0]}")
