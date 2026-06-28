import os
# Ensure matplotlib uses a non-GUI backend to avoid Tkinter image cleanup errors
os.environ.setdefault('MPLBACKEND', 'Agg')

import logging
import pandas as pd

from src.predict.predictor import Predictor

logging.basicConfig(level=logging.INFO)


def main(predict_type, input_path=None, input_array=None):
    """Package-level main for prediction workflows.
    Keeps the same signature used by `run_job.py`.
    """
    logging.info("\n" + "#" * 60)
    logging.info("# IRIS MLOPS - PREDICTION PIPELINE")
    logging.info("#" * 60)
    logging.info(f"Mode: {predict_type}")
    try:
        predictor = Predictor()

        if predict_type == 'batch_predict':
            if input_path is None:
                logging.error("CSV path required for batch_predict")
                raise ValueError("CSV path not provided")

            logging.info(f"Loading batch data from: {input_path}")
            # Detect headerless CSV similar to training data (header=None)
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

            feature_cols = df.columns[:4]
            X = df[feature_cols].values
            logging.info(f"Loaded {len(X)} samples with {X.shape[1]} features")

            y = None
            if len(df.columns) > 4:
                y = df[df.columns[4]].values
                logging.info("Target column found for evaluation")

            result = predictor.run_batch_prediction(X)

            if y is not None:
                logging.info("\nEvaluating predictions...")
                predictor.load_model()
                eval_result = predictor.evaluate(X, y)
                logging.info("Evaluation Metrics:")
                logging.info(f"  Accuracy:  {eval_result['accuracy']:.4f}")
                logging.info(f"  Precision: {eval_result['precision']:.4f}")
                logging.info(f"  Recall:    {eval_result['recall']:.4f}")
                logging.info(f"  F1 Score:  {eval_result['f1']:.4f}")
                result['metrics'] = eval_result

        elif predict_type == 'single_predict':
            if input_array is None:
                input_array = [5.1, 3.5, 1.4, 0.2]
                logging.info(f"Using default input sample: {input_array}")
            result = predictor.run_single_prediction(input_array)
            logging.info(f"Result: {result}")

        else:
            logging.error(f"Unknown predict type: {predict_type}")
            raise ValueError(f"Invalid predict_type: {predict_type}")

        logging.info("\n# PIPELINE EXECUTION COMPLETE")
        logging.info("#" * 60 + "\n")
        return result
    except Exception as e:
        logging.error(f"\n# PIPELINE EXECUTION FAILED: {e}")
        logging.error("#" * 60 + "\n")
        raise


if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        print("\nUsage:")
        print("  python -m src.predict single_predict [feature1 feature2 feature3 feature4]")
        print("  python -m src.predict batch_predict <csv_file_path>")
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
