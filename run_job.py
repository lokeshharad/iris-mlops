import argparse
import os

from src.train import main as train_main
from src.predict import main as predict_main

def parse_args():
    parser = argparse.ArgumentParser(description="Run a job with specified parameters.")
    parser.add_argument("-op", "--operation", type=str, choices=["train", "predict"], required=True, help="The operation to perform (e.g., train, predict).")
    parser.add_argument('-inp', '--input_array', type=str, required=False, help="Input array for prediction as string representation (e.g., '[5.1, 3.5, 1.4, 0.2]').")

    return parser.parse_args()

def train_operation(args):
    print("Training model")
    train_main()

def predict_operation(args):
    print(f"Making predictions with input array from: {args.input_array}")
    input_array = eval(args.input_array)
    predict_main(input_array)

operations = {
    "train": train_operation,
    "predict": predict_operation
}

def main():
    args = parse_args()
    operations[args.operation](args)

main()