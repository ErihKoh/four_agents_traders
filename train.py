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
    except FileNotFoundError:
        logging.error(f"Файл {data_path} не знайдено!")
        return

    trainer = ModelTrainer(config)
    # Тепер отримуємо словник метрик
    metrics = trainer.train(df)

    # Витягуємо Accuracy для порівняння
    accuracy = metrics['Accuracy']

    if accuracy > 0.52:
        logging.info(f"🔥 ПЕРЕМОГА! Точність {accuracy:.4f} вища за поріг. Модель збережена.")
    else:
        logging.warning(f"📉 Точність {accuracy:.4f} замала для реальної торгівлі.")


if __name__ == "__main__":
    main()