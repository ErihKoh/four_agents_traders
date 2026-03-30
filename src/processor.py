import pandas as pd
import numpy as np
import os
import joblib


class FeatureEngineer:
    def __init__(self, config):
        self.config = config
        self.models_dir = config['paths']['models_dir']
        # Розрахунок горизонту: 20 хв / таймфрейм
        target_min = config['trading'].get('target_minutes', 20)
        tf_min = config['trading'].get('timeframe_minutes', 1)
        self.horizon = max(1, target_min // tf_min)

    def process(self, df, is_training=True):
        df = df.copy()
        df.columns = [c.lower() for c in df.columns]

        # 1. ТЕХНІЧНІ ФІЧІ (Чиста математика без часу)
        # Відстань від EMA (головна фіча з Mac)
        df['ema50'] = df['close'].ewm(span=50, adjust=False).mean()
        df['dist_ema50'] = (df['close'] - df['ema50']) / df['close'] * 100

        # RSI (класика)
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-10)
        df['rsi'] = 100 - (100 / (1 + rs))

        # Моментум та волатильність
        df['mom_5'] = df['close'].diff(5)
        df['volatility'] = df['close'].rolling(20).std()

        # 2. PRICE ACTION ПАТЕРНИ (Те, що ти хотів додати)
        body = (df['close'] - df['open']).abs()
        high_wick = df['high'] - df[['open', 'close']].max(axis=1)
        low_wick = df[['open', 'close']].min(axis=1) - df['low']

        # Pin Bar (хвіст у 2 рази більший за тіло)
        df['pin_bar_up'] = ((low_wick > body * 2) & (high_wick < body * 0.5)).astype(int)
        df['pin_bar_down'] = ((high_wick > body * 2) & (low_wick < body * 0.5)).astype(int)

        # Engulfing (Поглинання)
        df['eng_up'] = (
                    (df['close'] > df['open']) & (body > body.shift(1)) & (df['close'] > df['high'].shift(1))).astype(
            int)
        df['eng_down'] = (
                    (df['close'] < df['open']) & (body > body.shift(1)) & (df['close'] < df['low'].shift(1))).astype(
            int)

        # 3. TARGET (Правильний Shift)
        if is_training:
            # Чи буде ціна через 20 хв вище ніж зараз?
            df['target'] = (df['close'].shift(-self.horizon) > df['close']).astype(int)
            df.dropna(inplace=True)

        # 4. CLEANUP (Видаляємо ВСЕ часове та сирі ціни)
        # Ми прибираємо 'hour_sin', 'is_london' тощо, щоб модель не ставала "годинником"
        features_to_keep = [
            'time', 'target', 'rsi', 'dist_ema50', 'mom_5', 'volatility',
            'pin_bar_up', 'pin_bar_down', 'eng_up', 'eng_down'
        ]

        df = df[[c for c in df.columns if c in features_to_keep]]
        df = df.dropna().reset_index(drop=True)

        return df