import argparse
import os
import logging

from src.train import main as train_main
from src.predict import main as predict_main

logging.basicConfig(level=logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser(description="Run training or prediction jobs.")
    parser.add_argument("-op", "--operation", required=True, choices=["train", "single_predict", "batch_predict"],
                        help="Operation to perform: train | single_predict | batch_predict")

    # Single prediction: accept 4 numeric features
    parser.add_argument('--features', nargs=4, type=float,
                        help="Four features for single prediction: sepal_length sepal_width petal_length petal_width")

    # Batch prediction: CSV path
    parser.add_argument('--csv', type=str, help="Path to CSV file for batch prediction")

    return parser.parse_args()


def train_handler(args):
    logging.info("Starting training operation")
    train_main()


def single_predict_handler(args):
    logging.info("Starting single prediction operation")
    if not args.features:
        logging.error("--features is required for single_predict (4 numeric values)")
        raise SystemExit(1)
    features = list(args.features)
    logging.info(f"Features: {features}")
    predict_main('single_predict', input_array=features)


def batch_predict_handler(args):
    logging.info("Starting batch prediction operation")
    if not args.csv:
        logging.error("--csv is required for batch_predict (path to CSV file)")
        raise SystemExit(1)
    csv_path = args.csv
    logging.info(f"CSV path: {csv_path}")
    predict_main('batch_predict', input_path=csv_path)


OPERATIONS = {
    'train': train_handler,
    'single_predict': single_predict_handler,
    'batch_predict': batch_predict_handler,
}


def main():
    args = parse_args()
    handler = OPERATIONS.get(args.operation)
    if handler is None:
        logging.error(f"Unsupported operation: {args.operation}")
        raise SystemExit(1)
    # dispatch to the handler based solely on the argument value
    handler(args)


if __name__ == '__main__':
    main()