import pandas as pd
import numpy as np
import os
import joblib
from sklearn.preprocessing import MinMaxScaler


class FeatureEngineer:
    def __init__(self, config):
        self.config = config
        self.models_dir = config['paths']['models_dir']
        self.horizon = config['trading'].get('horizon', 60)
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self._scaler_cols = None

    def process(self, df, is_training=True):
        df = df.copy()
        df.columns = [c.lower() for c in df.columns]
        df = df.sort_values('time').reset_index(drop=True)

        # --- 1. ТЕХНІЧНІ ІНДИКАТОРИ M1 ---
        df['ema20'] = df['close'].ewm(span=20, adjust=False).mean()
        df['ema50'] = df['close'].ewm(span=50, adjust=False).mean()

        # RSI 14
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        df['rsi'] = 100 - (100 / (1 + gain / (loss + 1e-10)))

        # RSI 7 та RSI 21
        for period in [7, 21]:
            d = df['close'].diff()
            g = d.where(d > 0, 0).rolling(period).mean()
            l = (-d.where(d < 0, 0)).rolling(period).mean()
            df[f'rsi_{period}'] = 100 - (100 / (1 + g / (l + 1e-10)))

        # ATR
        df['atr'] = (df['high'] - df['low']).rolling(14).mean() * 1000

        # MACD
        ema12 = df['close'].ewm(span=12, adjust=False).mean()
        ema26 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = (ema12 - ema26) * 1000
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']

        # Bollinger Bands
        bb_period = 20
        bb_std = df['close'].rolling(bb_period).std()
        bb_mid = df['close'].rolling(bb_period).mean()
        df['bb_width'] = (bb_std * 2) / (bb_mid + 1e-10) * 1000
        df['bb_position'] = (
            (df['close'] - (bb_mid - bb_std * 2)) /
            (bb_std * 4 + 1e-10)
        )

        # --- 2. ТЕХНІЧНІ ІНДИКАТОРИ M5 ---
        if 'close_m5' in df.columns:
            df['ema20_m5'] = df['close_m5'].ewm(span=20, adjust=False).mean()
            df['dist_m1_ema20_m5'] = (df['close'] - df['ema20_m5']) * 1000
            df['dist_m1_m5'] = (df['close'] - df['close_m5']) * 1000
            df['momentum_m5'] = df['close_m5'].diff(3) * 1000
            df['volatility_m5'] = df['close_m5'].rolling(10).std() * 1000

        # --- 3. ВІДНОСНІ ФІЧІ ---
        df['dist_ema20'] = (df['close'] - df['ema20']) * 1000
        df['dist_ema50'] = (df['close'] - df['ema50']) * 1000
        df['body_size'] = (df['close'] - df['open']) * 1000

        # Velocity нормалізована по ATR
        atr_safe = df['atr'].replace(0, 1)
        df['velocity_5m'] = (df['close'].diff(5) * 1000) / atr_safe
        df['velocity_15m'] = (df['close'].diff(15) * 1000) / atr_safe

        # Momentum
        df['momentum_10'] = df['close'].diff(10) * 1000
        df['momentum_20'] = df['close'].diff(20) * 1000

        # Volatility
        df['volatility_20'] = df['close'].rolling(20).std() * 1000
        df['volatility_ratio'] = (
            df['volatility_20'] /
            (df['close'].rolling(60).std() * 1000 + 1e-10)
        )

        # --- 4. СВІЧКОВИЙ ПАТЕРН ---
        high_body = df[['open', 'close']].max(axis=1)
        low_body = df[['open', 'close']].min(axis=1)
        df['upper_shadow'] = (df['high'] - high_body) * 1000
        df['lower_shadow'] = (low_body - df['low']) * 1000
        df['shadow_ratio'] = (
            df['upper_shadow'] / (df['lower_shadow'] + 1e-10)
        )

        # Серія напрямків (streak)
        df['direction'] = np.sign(df['close'] - df['open'])
        streak_group = (df['direction'] != df['direction'].shift()).cumsum()
        df['streak'] = (
            df.groupby(streak_group)['direction'].cumcount() + 1
        ) * df['direction']

        # --- 5. ЧАС ДНЯ ТА СЕСІЇ ---
        df['hour'] = df['time'].dt.hour
        df['weekday'] = df['time'].dt.dayofweek

        # Циклічне кодування часу (краще ніж просто число)
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['minute_sin'] = np.sin(2 * np.pi * df['time'].dt.minute / 60)
        df['minute_cos'] = np.cos(2 * np.pi * df['time'].dt.minute / 60)
        df['weekday_sin'] = np.sin(2 * np.pi * df['weekday'] / 5)
        df['weekday_cos'] = np.cos(2 * np.pi * df['weekday'] / 5)

        # Торгові сесії
        df['is_london'] = ((df['hour'] >= 8) & (df['hour'] < 16)).astype(int)
        df['is_ny'] = ((df['hour'] >= 13) & (df['hour'] < 21)).astype(int)
        df['is_overlap'] = ((df['hour'] >= 13) & (df['hour'] < 16)).astype(int)
        df['is_asian'] = ((df['hour'] >= 0) & (df['hour'] < 8)).astype(int)
        df['is_monday'] = (df['weekday'] == 0).astype(int)
        df['is_friday'] = (df['weekday'] == 4).astype(int)

        # --- 6. ТАРГЕТ ---
        if is_training:
            df['target'] = (
                df['close'].shift(-self.horizon) > df['close']
            ).astype(int)
            df = df.dropna(subset=['target'])

        # --- 7. ВИДАЛЕННЯ АБСОЛЮТНИХ ЦІН ---
        exact_forbidden = {
            'open', 'high', 'low', 'close',
            'open_m5', 'high_m5', 'low_m5', 'close_m5',
            'ema20', 'ema50', 'ema20_m5', 'atr',
            'hour', 'weekday',  # замінені циклічним кодуванням
        }
        cols_to_drop = [c for c in df.columns if c in exact_forbidden]
        df = df.drop(columns=cols_to_drop, errors='ignore')

        df = df.dropna().reset_index(drop=True)

        return self._scale(df, is_training)

    def _scale(self, df, is_training):
        exclude = ['time', 'target']
        scaler_path = os.path.join(self.models_dir, 'scaler.pkl')
        cols_path = os.path.join(self.models_dir, 'scaler_cols.pkl')

        if is_training:
            cols_to_scale = [c for c in df.columns if c not in exclude]
            if df.empty:
                return df
            self._scaler_cols = cols_to_scale
            df[cols_to_scale] = self.scaler.fit_transform(df[cols_to_scale])
            os.makedirs(self.models_dir, exist_ok=True)
            joblib.dump(self.scaler, scaler_path)
            joblib.dump(cols_to_scale, cols_path)
        else:
            if not os.path.exists(scaler_path) or not os.path.exists(cols_path):
                raise FileNotFoundError(
                    "Скалер не знайдено. Спочатку запустіть train.py"
                )
            self.scaler = joblib.load(scaler_path)
            expected_cols = joblib.load(cols_path)
            self._scaler_cols = expected_cols

            for col in expected_cols:
                if col not in df.columns:
                    df[col] = 0.0

            # Трансформуємо в точному порядку як при навчанні
            df[expected_cols] = self.scaler.transform(df[expected_cols])

        return df