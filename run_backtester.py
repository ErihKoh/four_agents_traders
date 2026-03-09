import yaml
import joblib
import pandas as pd
import logging
from src.processor import FeatureEngineer
from src.strategy import MLStrategy
from src.backtester import VectorizedBacktester


def main():
    logging.basicConfig(level=logging.INFO)
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # 1. Завантаження моделі та сирих даних
    try:
        model = joblib.load(f"{config['paths']['models_dir']}/xgb_baseline.pkl")
        df_raw = pd.read_parquet(f"data/{config['trading']['symbol']}_ml_ready.parquet")
    except Exception as e:
        logging.error(f"Помилка завантаження: {e}")
        return

    # 2. Обробка фіч для моделі
    processor = FeatureEngineer(config)
    df_features = processor.process(df_raw, is_training=False)

    # 3. Синхронізація сирих цін з обробленими фічами
    # Processor міг видалити рядки (через dropna), тому ми беремо ціни
    # тільки для тих моментів часу, які залишилися в df_features
    df_prices = df_raw[df_raw['time'].isin(df_features['time'])].copy()

    # 4. Відрізаємо Out-of-Sample (тестову частину)
    split_idx = int(len(df_features) * (1 - config['model']['test_size']))

    test_features = df_features.iloc[split_idx:].copy()
    test_prices = df_prices.iloc[split_idx:].copy()

    # 5. Генерація сигналів
    strategy = MLStrategy(config)
    signals, probs = strategy.generate_signals(test_features, model)

    # 6. Запуск бектесту (передаємо ціни, а не фічі!)
    backtester = VectorizedBacktester(config)
    metrics, results = backtester.run(test_prices, signals)

    print("\n" + "=" * 45)
    print(f"💰 РЕЗУЛЬТАТИ СИМУЛЯЦІЇ (Threshold: {config['trading']['threshold']})")
    print("=" * 45)
    for k, v in metrics.items():
        print(f"{k:20}: {v}")
    print("=" * 45)


if __name__ == "__main__":
    main()