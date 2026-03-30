import yaml
import logging
import pandas as pd
import os
from src.processor import FeatureEngineer
from src.model_trainer import ModelTrainer


def main():
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    # 1. Завантаження конфігурації
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # 2. Підготовка даних (MacBook Logic)
    # Використовуємо наш новий процесор, який рахує Shift та Price Action
    data_path = config['paths']['data_path']
    try:
        raw_df = pd.read_parquet(data_path)
        logging.info(f"Завантажено {len(raw_df)} рядків")

        engineer = FeatureEngineer(config)
        # is_training=True активує створення 'target' з правильним shift
        df = engineer.process(raw_df, is_training=True)
        logging.info(f"Дані оброблені. Фінальний розмір: {len(df)}")
    except Exception as e:
        logging.error(f"Помилка завантаження/обробки: {e}")
        return

    # 3. Список фіч (Тільки ціна та патерни, НІЯКОГО ЧАСУ)
    FEATURES = [
        'rsi', 'dist_ema50', 'mom_5', 'volatility',
        'pin_bar_up', 'pin_bar_down', 'eng_up', 'eng_down'
    ]

    # Залишаємо тільки потрібні колонки для навчання
    train_df = df[FEATURES + ['target']]

    # 4. Навчання
    trainer = ModelTrainer(config)
    # Передаємо оброблені дані
    metrics = trainer.train(train_df)

    # 5. Перевірка результату
    accuracy = metrics.get('Accuracy', 0)

    # На MacBook ми вважали 52% вже робочим результатом для M1
    if accuracy > 0.52:
        logging.info(f"✅ MacBook Style OK: Accuracy={accuracy:.4f}")
    else:
        logging.warning(f"⚠️ Слабкий сигнал: Accuracy={accuracy:.4f}.")


if __name__ == "__main__":
    main()