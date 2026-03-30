import yaml
import joblib
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.processor import FeatureEngineer
from src.strategy import MLStrategy
from src.backtester import VectorizedBacktester


def main():
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    try:
        model = joblib.load(f"{config['paths']['models_dir']}/xgb_baseline.pkl")
        df_raw = pd.read_parquet(f"data/{config['trading']['symbol']}_ml_ready.parquet")
    except Exception as e:
        logging.error(f"Помилка завантаження: {e}")
        return

    processor = FeatureEngineer(config)
    df_features = processor.process(df_raw, is_training=False)

    df_prices = df_raw[df_raw['time'].isin(df_features['time'])].copy()

    split_time = df_features['time'].iloc[int(len(df_features) * (1 - config['model']['test_size']))]
    test_features = df_features[df_features['time'] >= split_time].copy()
    test_prices = df_prices[df_prices['time'] >= split_time].copy()

    if len(test_features) != len(test_prices):
        logging.error(f"Розмір features ({len(test_features)}) не збігається з prices ({len(test_prices)}).")
        return

    strategy = MLStrategy(config)
    signals, probs = strategy.generate_signals(test_features, model)

    # =============================================
    # 🔍 ДІАГНОСТИКА
    # =============================================
    print("\n" + "=" * 45)
    print("🔍 ДІАГНОСТИКА МОДЕЛІ")
    print("=" * 45)
    print(f"  Середня prob:       {probs.mean():.4f}")
    print(f"  Медіана prob:       {np.median(probs):.4f}")
    print(f"  Мін / Макс prob:    {probs.min():.4f} / {probs.max():.4f}")
    print(f"  Сигналів BUY:       {(signals == 1).sum()}")
    print(f"  Сигналів SELL:      {(signals == -1).sum()}")
    print(f"  Сигналів WAIT:      {(signals == 0).sum()}")
    print(f"  Всього свічок:      {len(signals)}")
    print("=" * 45)

    # Топ фічі
    feature_cols = test_features.drop(columns=['time', 'target'], errors='ignore').columns
    importances = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    print("\n🏆 ТОП-7 ФІЧ:")
    print(importances.head(7).to_string(index=False))
    print("=" * 45)

    # Графік розподілу ймовірностей
    import os
    os.makedirs("logs", exist_ok=True)
    plt.figure(figsize=(10, 4))
    plt.hist(probs, bins=50, color='steelblue', edgecolor='white')
    plt.axvline(config['trading']['threshold'], color='red',
                linestyle='--', label=f"Threshold {config['trading']['threshold']}")
    plt.axvline(1 - config['trading']['threshold'], color='orange',
                linestyle='--', label=f"1-Threshold {1 - config['trading']['threshold']}")
    plt.title("Розподіл ймовірностей моделі (тестова вибірка)")
    plt.xlabel("Probability UP")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig("logs/prob_distribution.png")
    print("\n📊 Графік збережено: logs/prob_distribution.png")
    # =============================================

    backtester = VectorizedBacktester(config)
    metrics, results = backtester.run(test_prices.reset_index(drop=True), signals)

    print("\n" + "=" * 45)
    print(f"💰 РЕЗУЛЬТАТИ БЕКТЕСТУ (Threshold: {config['trading']['threshold']})")
    print("=" * 45)
    for k, v in metrics.items():
        print(f"  {k:20}: {v}")
    print("=" * 45)


if __name__ == "__main__":
    main()