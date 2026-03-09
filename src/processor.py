import pandas as pd
import numpy as np
import os
import joblib
from sklearn.preprocessing import MinMaxScaler


class FeatureEngineer:
    def __init__(self, config):
        self.config = config
        self.models_dir = config['paths']['models_dir']
        self.horizon = config['trading'].get('horizon', 20)
        self.scaler = MinMaxScaler(feature_range=(0, 1))

    def process(self, df, is_training=True):
        df = df.copy()

        # 0. НОРМАЛІЗАЦІЯ РЕГІСТРУ (Fix для помилок "unseen feature names")
        # Переводимо всі назви колонок у нижній регістр відразу
        df.columns = [c.lower() for c in df.columns]

        df = df.sort_values('time').reset_index(drop=True)

        # --- 1. ТЕХНІЧНІ ІНДИКАТОРИ (M1) ---
        df['ema20'] = df['close'].ewm(span=20, adjust=False).mean()
        df['ema50'] = df['close'].ewm(span=50, adjust=False).mean()

        # RSI (M1)
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))

        # --- 2. ТЕХНІЧНІ ІНДИКАТОРИ (M5) ---
        # Перевіряємо наявність m5 даних (також у нижньому регістрі)
        if 'close_m5' in df.columns:
            df['ema20_m5'] = df['close_m5'].ewm(span=20, adjust=False).mean()
            # Відстань між ціною M1 та середньою M5
            df['dist_m1_ema20_m5'] = (df['close'] - df['ema20_m5']) * 1000
            # Різниця між M1 close та M5 close
            df['dist_m1_m5'] = (df['close'] - df['close_m5']) * 1000

        # --- 3. ВІДНОСНІ ФІЧІ ---
        df['dist_ema20'] = (df['close'] - df['ema20']) * 1000
        df['dist_ema50'] = (df['close'] - df['ema50']) * 1000
        df['body_size'] = (df['close'] - df['open']) * 1000
        df['velocity_5m'] = df['close'].diff(5) * 1000
        df['velocity_15m'] = df['close'].diff(15) * 1000

        # --- 4. СТВОРЕННЯ ТАРГЕТУ ---
        if is_training:
            df['target'] = (df['close'].shift(-self.horizon) > df['close']).astype(int)
            df = df.dropna(subset=['target'])

        # --- 5. ЖОРСТКА ЗАЧИСТКА АБСОЛЮТНИХ ЦІН ТА ОБ'ЄМІВ ---
        # Видаляємо все, що містить ціну або об'єм, щоб модель не "зубрила" числа
        # Тепер ми шукаємо ключові слова в нижньому регістрі
        forbidden = ['open', 'high', 'low', 'close', 'vol', 'ema']

        cols_to_drop = []
        for col in df.columns:
            # Видаляємо, якщо назва містить заборонене слово...
            if any(f in col for f in forbidden):
                # ...АЛЕ залишаємо наші розраховані дистанції та швидкості
                if not col.startswith('dist_') and not col.startswith('velocity_'):
                    cols_to_drop.append(col)

        df = df.drop(columns=cols_to_drop, errors='ignore')

        # Видаляємо NaN та скидаємо індекс
        df = df.dropna().reset_index(drop=True)

        return self._scale(df, is_training)

    def _scale(self, df, is_training):
        exclude = ['time', 'target']
        cols_to_scale = [c for c in df.columns if c not in exclude]

        scaler_path = os.path.join(self.models_dir, 'scaler.pkl')

        if is_training:
            if not df.empty:
                df[cols_to_scale] = self.scaler.fit_transform(df[cols_to_scale])
                os.makedirs(self.models_dir, exist_ok=True)
                joblib.dump(self.scaler, scaler_path)
        else:
            if os.path.exists(scaler_path):
                self.scaler = joblib.load(scaler_path)
                for col in cols_to_scale:
                    if col not in df.columns:
                        df[col] = 0
                df[cols_to_scale] = self.scaler.transform(df[cols_to_scale])

        return df