import pandas as pd
import logging
from io import StringIO

class DataPipeline:
    def __init__(self, config):
        self.symbol = config['trading']['symbol']
        self.logger = logging.getLogger("Bot.DataPipeline")

    def parse_combined_data(self, raw_string):
        """Розділяє, парсить та синхронізує M1/M5"""
        if not raw_string or "ERROR" in raw_string:
            return None

        parts = raw_string.split("|M5_SEPARATOR|")
        if len(parts) < 2:
            self.logger.warning("M5 не знайдено, парсимо тільки M1")
            return self._parse_csv(parts[0])

        # Парсинг та валідація окремо
        df_m1 = self.validate_data(self._parse_csv(parts[0]), "M1")
        df_m5 = self.validate_data(self._parse_csv(parts[1]), "M5")

        if df_m1 is None or df_m5 is None: return None
        return self.sync_timeframes(df_m1, df_m5)

    def _parse_csv(self, csv_str):
        """Універсальний парсер блоку CSV"""
        clean_str = "\n".join([l for l in csv_str.strip().split('\n') if l.strip() not in ["M1", "M5"]])
        try:
            df = pd.read_csv(StringIO(clean_str), on_bad_lines='skip')
            df.columns = [c.strip().lower() for c in df.columns]
            df['time'] = pd.to_datetime(df['time'], format='mixed', errors='coerce')
            return df.dropna(subset=['time']).sort_values('time')
        except Exception as e:
            self.logger.error(f"Помилка CSV: {e}")
            return None

    def validate_data(self, df, label):
        if df is None or df.empty: return None
        initial = len(df)
        df = df.drop_duplicates(subset=['time']).dropna()
        if len(df) < initial:
            self.logger.warning(f"[{label}] Очищено {initial - len(df)} рядків")
        return df

    def sync_timeframes(self, df_m1, df_m5):
        """Point-in-Time Join (Без витоку даних)"""
        df_m5 = df_m5.rename(columns={c: f"{c}_m5" for c in df_m5.columns if c != 'time'})
        df_combined = pd.merge_asof(df_m1, df_m5, on='time', direction='backward')
        return df_combined.dropna().reset_index(drop=True)

    def save_parquet(self, df, suffix=""):
        path = f"data/{self.symbol}_{suffix}.parquet"
        df.to_parquet(path, index=False)
        return path