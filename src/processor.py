# MVP 2: Feature Engineering

import pandas as pd
import numpy as np
import logging


class FeatureEngineer:
    def __init__(self, horizon=20):
        """
        :param horizon: Кількість свічок у майбутнє для прогнозу (Target)
        """
        self.horizon = horizon
        self.logger = logging.getLogger("Bot.Processor")

    def add_indicators(self, df):
        """MVP 2: Розрахунок технічних індикаторів"""
        # Створюємо копію, щоб не псувати вхідні дані
        df = df.copy()

        # 1. EMA (Trend) - визначають напрямок тренду
        df['ema20'] = df['close'].ewm(span=20, adjust=False).mean()
        df['ema50'] = df['close'].ewm(span=50, adjust=False).mean()
        df['ema200'] = df['close'].ewm(span=200, adjust=False).mean()

        # 2. RSI (Momentum) - показує перекупленість/перепроданість
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))

        # 3. ATR (Volatility) - показує середній діапазон руху (для стоп-лоссів)
        high_low = df['high'] - df['low']
        df['atr'] = high_low.rolling(window=14).mean()

        # 4. Похідні фічі (Математика)
        df['log_return'] = np.log(df['close'] / df['close'].shift(1))
        # Відстань ціни від "важкої" середньої (показує аномальні відхилення)
        df['dist_ema200'] = (df['close'] - df['ema200']) / df['ema200']

        return df

    def add_labels(self, df):
        """MVP 3: Формування цільової змінної (Labeling)"""
        # Зсуваємо ціну закриття на N свічок назад, щоб отримати "майбутню ціну"
        df['future_close'] = df['close'].shift(-self.horizon)

        # 1 - якщо ціна виросла через N свічок, 0 - якщо впала або залишилась такою ж
        df['target'] = (df['future_close'] > df['close']).astype(int)

        # Видаляємо тимчасову колонку
        df = df.drop(columns=['future_close'])

        return df

    def process(self, df):
        """Повний цикл обробки"""
        self.logger.info(f"Обробка датасету: {len(df)} рядків...")

        df = self.add_indicators(df)
        df = self.add_labels(df)

        # КРИТИЧНО: Видаляємо всі рядки з NaN
        # Це автоматично прибере перші 200 (через EMA200) і останні N (через Target)
        df_final = df.dropna()

        self.logger.info(f"Обробка завершена. Залишилось чистих рядків: {len(df_final)}")
        return df_final