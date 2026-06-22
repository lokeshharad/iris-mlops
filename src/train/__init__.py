import logging
import os

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

        trainer = Trainer(data_path=data_path, model_path=model_path)
        trainer.run()
        logging.info("\n# PIPELINE EXECUTION COMPLETE")
        logging.info("#" * 60 + "\n")
    except Exception as e:
        logging.error(f"\n# PIPELINE EXECUTION FAILED: {e}")
        logging.error("#" * 60 + "\n")
        raise


if __name__ == '__main__':
    main()
