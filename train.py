import yaml
import logging
import pandas as pd
from src.model_trainer import ModelTrainer


def main():
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    data_path = f"data/{config['trading']['symbol']}_ml_ready.parquet"
    try:
        df = pd.read_parquet(data_path)
        logging.info(f"Завантажено {len(df)} рядків з {data_path}")
    except FileNotFoundError:
        logging.error(f"Файл {data_path} не знайдено. Спочатку зберіть дані.")
        return

    trainer = ModelTrainer(config)
    metrics = trainer.train(df)

    accuracy = metrics.get('Accuracy', 0)
    roc_auc = metrics.get('ROC-AUC', 0)

    if accuracy > 0.52 and roc_auc > 0.55:
        logging.info(f"✅ Модель пройшла поріг: Accuracy={accuracy:.4f}, ROC-AUC={roc_auc:.4f}")
    else:
        logging.warning(
            f"⚠️ Модель не досягла порогу: Accuracy={accuracy:.4f}, ROC-AUC={roc_auc:.4f}. "
            f"Розгляньте збільшення даних або зміну гіперпараметрів."
        )


if __name__ == "__main__":
    main()