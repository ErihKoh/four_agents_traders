import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib
import os


class FeatureEngineer:
    def __init__(self, config):
        self.horizon = config['trading']['horizon']
        self.models_dir = config['paths']['models_dir']
        self.scaler = MinMaxScaler()

    def process(self, df, is_training=True):
        df = df.copy()

        # 1. Розрахунок індикаторів (завжди потрібні)
        for tf in ['', '_m5']:
            suffix = 'm1' if tf == '' else 'm5'
            close = df[f'close{tf}']
            df[f'ema20_{suffix}'] = close.ewm(span=20, adjust=False).mean()
            df[f'rsi_{suffix}'] = self._rsi(close)

        # 2. Крос-ТФ та математика
        df['rsi_diff'] = df['rsi_m1'] - df['rsi_m5']
        df['log_ret'] = np.log(df['close'] / df['close'].shift(1))

        # 3. ЛОГІКА ТАРГЕТУ (Тут була помилка)
        if is_training:
            # Створюємо таргет тільки для навчання
            df['target'] = (df['close'].shift(-self.horizon) > df['close']).astype(int)
            # Видаляємо NaN на початку (через індикатори) та в кінці (через таргет)
            df = df.dropna().reset_index(drop=True)
        else:
            # В режимі LIVE нам не потрібен таргет.
            # Видаляємо тільки рядки на початку, де індикатори ще "гріються" (warm-up)
            # Останній рядок ОБОВ'ЯЗКОВО залишається для прогнозу.
            df = df.dropna(subset=['ema20_m1', 'rsi_m1']).reset_index(drop=True)

        return self._scale(df, is_training)

    def _rsi(self, series, period=14):
        delta = series.diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = -delta.where(delta < 0, 0).rolling(period).mean()
        return 100 - (100 / (1 + (gain / loss)))

    def _scale(self, df, is_training):
        # Список колонок для скалювання (виключаємо час та таргет, якщо він є)
        exclude = ['time', 'target']
        cols = [c for c in df.columns if c not in exclude]
        scaler_path = os.path.join(self.models_dir, 'scaler.pkl')

        if is_training:
            df[cols] = self.scaler.fit_transform(df[cols])
            joblib.dump(self.scaler, scaler_path)
        else:
            # В режимі Live завантажуємо вже навчений скалер
            if os.path.exists(scaler_path):
                self.scaler = joblib.load(scaler_path)
                df[cols] = self.scaler.transform(df[cols])
            else:
                raise FileNotFoundError("Scaler не знайдено! Спочатку навчіть модель.")
        return df