import logging
import os

import mlflow
import mlflow.sklearn

from src.train.trainer import Trainer
from src.reference import DATA_FOLDER, MODEL_FOLDER, DATA_FILE, MODEL_FILE

logging.basicConfig(level=logging.INFO)


def main():
    logging.info("\n" + "#" * 60)
    logging.info("# IRIS MLOPS - MODEL TRAINING PIPELINE")
    logging.info("#" * 60)
    try:
        data_path = os.path.join(DATA_FOLDER, DATA_FILE)
        model_path = os.path.join(MODEL_FOLDER, MODEL_FILE)
        logging.info(f"Configuration: DATA_FOLDER={DATA_FOLDER}, MODEL_FOLDER={MODEL_FOLDER}")

        mlflow.set_experiment("iris-mlops")
        mlflow.sklearn.autolog()
        with mlflow.start_run(run_name="iris-training"):
            mlflow.log_param("data_path", data_path)
            mlflow.log_param("model_path", model_path)
            trainer = Trainer(data_path=data_path, model_path=model_path)
            mlflow.log_param("cv_splits", trainer.cv_splits)
            trainer.run()
        logging.info("\n# PIPELINE EXECUTION COMPLETE")
        logging.info("#" * 60 + "\n")
    except Exception as e:
        logging.error(f"\n# PIPELINE EXECUTION FAILED: {e}")
        logging.error("#" * 60 + "\n")
        raise


if __name__ == '__main__':
    main()
