import yaml
import joblib
import logging
import pandas as pd
import numpy as np
import os
from src.processor import FeatureEngineer
from src.strategy import MLStrategy, VirtualAccountant


def main():
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    # 1. Завантаження конфігурації
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # 2. Завантаження моделі та даних
    try:
        model = joblib.load(f"{config['paths']['models_dir']}/xgb_baseline.pkl")
        # Використовуємо шлях з конфігу
        df_raw = pd.read_parquet(config['paths']['data_path'])
        logging.info(f"Завантажено дані: {len(df_raw)} рядків")
    except Exception as e:
        logging.error(f"Помилка завантаження: {e}")
        return

    # 3. Обробка фіч
    processor = FeatureEngineer(config)
    df_features = processor.process(df_raw, is_training=False)

    # Синхронізація цін з фічами (щоб індекси збігалися)
    df_test = df_raw[df_raw['time'].isin(df_features['time'])].copy().reset_index(drop=True)
    df_features = df_features.reset_index(drop=True)

    # 4. Виділення тестової вибірки (Walk-Forward)
    split_idx = int(len(df_features) * (1 - config['model']['test_size']))
    test_features = df_features.iloc[split_idx:].copy().reset_index(drop=True)
    test_prices = df_test.iloc[split_idx:].copy().reset_index(drop=True)

    # 5. Генерація сигналів
    strategy = MLStrategy(config)
    signals, probs = strategy.generate_signals(test_features, model)

    # 6. Ініціалізація "Бухгалтера" (Емуляція торгівлі)
    accountant = VirtualAccountant(config)

    print("\n🚀 Початок симуляції...")

    # 1. Створюємо порожній список для результатів перед циклом
    trade_results = []

    # 2. Передаємо цей список у наш бухгалтер (Accountant)
    # Або просто збираємо результати після закриття

    print("\n🚀 Початок симуляції...")

    initial_balance = accountant.balance  # Запам'ятовуємо старт

    # Головний цикл: проходимо по кожній свічці
    for i in range(len(test_prices)):
        prev_balance = accountant.balance  # Баланс до перевірки угод

        current_price = test_prices.iloc[i]['close']
        current_vol = test_features.iloc[i]['volatility']

        accountant.check_pending(i, current_price)

        # Якщо баланс змінився — значить угода закрилася
        if accountant.balance != prev_balance:
            trade_results.append(accountant.balance - prev_balance)

        # Логіка відкриття (без змін)
        sig = signals[i]
        if sig != 0:
            accountant.open_trade(i, current_price, 'BUY' if sig == 1 else 'SELL', probs[i], current_vol)

    # 7. Фінальні результати
    print("\n" + "=" * 45)
    print(f"💰 РЕЗУЛЬТАТИ СИМУЛЯЦІЇ (MacBook Logic)")
    print("=" * 45)
    print(f"  Фінальний баланс : ${accountant.balance:.2f}")
    print(f"  Початковий баланс: ${config['backtest']['initial_balance']}")
    print(f"  Чистий прибуток  : ${accountant.balance - config['backtest']['initial_balance']:.2f}")
    print("=" * 45)

    # Діагностика фіч (як у тебе було)
    print("\n🏆 ТОП-5 ФІЧ МОДЕЛІ:")
    importances = pd.DataFrame({
        'feature': test_features.drop(columns=['time', 'target'], errors='ignore').columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    print(importances.head(5).to_string(index=False))

    print("\n" + "=" * 45)
    print(f"💰 РЕЗУЛЬТАТИ СИМУЛЯЦІЇ")
    print("=" * 45)

    wins = [r for r in trade_results if r > 0]
    losses = [abs(r) for r in trade_results if r < 0]

    pf = sum(wins) / sum(losses) if sum(losses) > 0 else 0

    print(f"  Фінальний баланс : ${accountant.balance:.2f}")
    print(f"  Кількість угод   : {len(trade_results)}")
    print(f"  Win Rate         : {len(wins) / len(trade_results) * 100:.1f}%" if trade_results else "0%")
    print(f"  Profit Factor    : {pf:.2f}")
    print("=" * 45)


if __name__ == "__main__":
    main()