import yaml
import joblib
import pandas as pd
import logging
from src.strategy import MLStrategy
from src.backtester import VectorizedBacktester


def main():
    logging.basicConfig(level=logging.INFO)

    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # 1. Завантаження моделі та даних
    try:
        model = joblib.load(f"{config['paths']['models_dir']}/xgb_baseline.pkl")
        df = pd.read_parquet(f"data/{config['trading']['symbol']}_ml_ready.parquet")
    except Exception as e:
        logging.error(f"Помилка завантаження: {e}")
        return

    # 2. Відрізаємо Out-of-Sample дані (на яких модель НЕ вчилася)
    split_idx = int(len(df) * (1 - config['model']['test_size']))
    test_df = df.iloc[split_idx:].copy()

    # 3. Емуляція торгівлі
    strategy = MLStrategy(config)
    signals = strategy.generate_signals(test_df, model)

    backtester = VectorizedBacktester(config)
    metrics, results = backtester.run(test_df, signals)

    print("\n" + "=" * 45)
    print(f"💰 РЕЗУЛЬТАТИ СИМУЛЯЦІЇ (Threshold: {config['trading']['threshold']})")
    print("=" * 45)
    for k, v in metrics.items():
        print(f"{k:20}: {v}")
    print("=" * 45)


if __name__ == "__main__":
    main()