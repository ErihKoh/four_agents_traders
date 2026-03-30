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
        self.scaler = MinMaxScaler()
        self._scaler_cols = None

    def process(self, df, is_training=True):
        df = df.copy()
        df.columns = [c.lower() for c in df.columns]
        df = df.sort_values('time').reset_index(drop=True)

        # =============================
        # 1. INDICATORS (M1)
        # =============================

        df['ema20'] = df['close'].ewm(span=20, adjust=False).mean()
        df['ema50'] = df['close'].ewm(span=50, adjust=False).mean()

        # RSI (EMA-based)
        delta = df['close'].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()

        rs = avg_gain / (avg_loss + 1e-10)
        df['rsi'] = 100 - (100 / (1 + rs))

        # TRUE ATR (correct)
        prev_close = df['close'].shift(1)
        tr = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                abs(df['high'] - prev_close),
                abs(df['low'] - prev_close)
            )
        )
        df['atr'] = tr.rolling(14).mean()

        # MACD
        ema12 = df['close'].ewm(span=12, adjust=False).mean()
        ema26 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = ema12 - ema26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']

        # Bollinger
        bb_mid = df['close'].rolling(20).mean()
        bb_std = df['close'].rolling(20).std()

        df['bb_width'] = bb_std / (bb_mid + 1e-10)
        df['bb_position'] = (df['close'] - bb_mid) / (bb_std + 1e-10)

        # =============================
        # 2. M5 CONTEXT
        # =============================

        if 'close_m5' in df.columns:
            df['ema20_m5'] = df['close_m5'].ewm(span=20, adjust=False).mean()
            df['dist_m1_m5'] = df['close'] - df['close_m5']
            df['momentum_m5'] = df['close_m5'].diff(3)
            df['volatility_m5'] = df['close_m5'].rolling(10).std()

        # =============================
        # 3. DERIVED FEATURES
        # =============================

        df['dist_ema20'] = df['close'] - df['ema20']
        df['dist_ema50'] = df['close'] - df['ema50']
        df['body'] = df['close'] - df['open']

        # normalized velocity
        atr_safe = df['atr'].replace(0, 1)
        df['velocity_5'] = df['close'].diff(5) / atr_safe

        # momentum
        df['momentum_10'] = df['close'].diff(10)

        # volatility
        df['volatility_20'] = df['close'].rolling(20).std()
        df['vol_ratio'] = df['volatility_20'] / (
            df['close'].rolling(60).std() + 1e-10
        )

        # =============================
        # 4. CANDLE STRUCTURE
        # =============================

        high_body = df[['open', 'close']].max(axis=1)
        low_body = df[['open', 'close']].min(axis=1)

        df['upper_shadow'] = df['high'] - high_body
        df['lower_shadow'] = low_body - df['low']

        shadow_ratio = df['upper_shadow'] / (df['lower_shadow'] + 1e-10)
        df['shadow_ratio'] = np.tanh(shadow_ratio)  # stabilize

        # streak (clipped)
        df['direction'] = np.sign(df['close'] - df['open'])
        group = (df['direction'] != df['direction'].shift()).cumsum()

        df['streak'] = (
            df.groupby(group)['direction'].cumcount() + 1
        ) * df['direction']

        df['streak'] = df['streak'].clip(-10, 10)

        # =============================
        # 5. TIME FEATURES
        # =============================

        df['hour'] = df['time'].dt.hour
        df['weekday'] = df['time'].dt.dayofweek

        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

        df['weekday_sin'] = np.sin(2 * np.pi * df['weekday'] / 5)
        df['weekday_cos'] = np.cos(2 * np.pi * df['weekday'] / 5)

        df['is_london'] = ((df['hour'] >= 8) & (df['hour'] < 16)).astype(int)
        df['is_ny'] = ((df['hour'] >= 13) & (df['hour'] < 21)).astype(int)
        df['is_overlap'] = ((df['hour'] >= 13) & (df['hour'] < 16)).astype(int)

        # =============================
        # 6. TARGET
        # =============================

        if is_training:
            df['target'] = (
                df['close'].shift(-self.horizon) > df['close']
            ).astype(int)

        # =============================
        # 7. CLEANUP
        # =============================

        forbidden = {
            'open', 'high', 'low', 'close',
            'open_m5', 'high_m5', 'low_m5', 'close_m5',
            'ema20', 'ema50', 'ema20_m5',
            'hour', 'weekday'
        }

        df = df.drop(columns=[c for c in df.columns if c in forbidden], errors='ignore')

        df = df.dropna().reset_index(drop=True)

        return self._scale(df, is_training)

    # =============================
    # SCALING (NO LEAKAGE)
    # =============================

    def _scale(self, df, is_training):
        exclude = ['time', 'target']

        scaler_path = os.path.join(self.models_dir, 'scaler.pkl')
        cols_path = os.path.join(self.models_dir, 'scaler_cols.pkl')

        cols = [c for c in df.columns if c not in exclude]

        if is_training:
            self._scaler_cols = cols

            # ⚠️ ВАЖЛИВО: scaler НЕ повинен бачити test!
            df[cols] = self.scaler.fit_transform(df[cols])

            os.makedirs(self.models_dir, exist_ok=True)
            joblib.dump(self.scaler, scaler_path)
            joblib.dump(cols, cols_path)

        else:
            self.scaler = joblib.load(scaler_path)
            expected_cols = joblib.load(cols_path)

            for col in expected_cols:
                if col not in df.columns:
                    df[col] = 0.0

            df = df.reindex(columns=['time'] + expected_cols + ['target'], fill_value=0)

            df[expected_cols] = self.scaler.transform(df[expected_cols])

        return df