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
            self.logger.error("Отримано порожні або помилкові дані від MT5.")
            return None

        parts = raw_string.split("|M5_SEPARATOR|")
        if len(parts) < 2:
            # M5 відсутній — критична помилка, модель без M5 фіч не працює
            self.logger.error("M5 дані відсутні. Пропускаємо пакет.")
            return None

        df_m1 = self.validate_data(self._parse_csv(parts[0]), "M1")
        df_m5 = self.validate_data(self._parse_csv(parts[1]), "M5")

        if df_m1 is None or df_m5 is None:
            return None

        return self.sync_timeframes(df_m1, df_m5)

    def _parse_csv(self, csv_str):
        """Універсальний парсер блоку CSV"""
        lines = csv_str.strip().split('\n')
        clean_lines = [l for l in lines if l.strip() not in ["M1", "M5", ""]]
        clean_str = "\n".join(clean_lines)

        try:
            df = pd.read_csv(StringIO(clean_str), on_bad_lines='warn')
            df.columns = [c.strip().lower() for c in df.columns]
            df['time'] = pd.to_datetime(df['time'], format='mixed', errors='coerce')
            return df.dropna(subset=['time']).sort_values('time').reset_index(drop=True)
        except Exception as e:
            self.logger.error(f"Помилка парсингу CSV: {e}")
            return None

    def validate_data(self, df, label):
        if df is None or df.empty:
            self.logger.error(f"[{label}] DataFrame порожній або None.")
            return None

        initial = len(df)
        df = df.drop_duplicates(subset=['time']).dropna()
        removed = initial - len(df)

        if removed > 0:
            self.logger.warning(f"[{label}] Очищено {removed} рядків (дублікати/NaN).")

        if len(df) < 50:
            self.logger.error(f"[{label}] Замало даних після очищення: {len(df)} рядків.")
            return None

        return df

    def sync_timeframes(self, df_m1, df_m5):
        """Point-in-Time Join (без витоку даних)"""
        df_m5_renamed = df_m5.rename(
            columns={c: f"{c}_m5" for c in df_m5.columns if c != 'time'}
        )
        df_combined = pd.merge_asof(
            df_m1, df_m5_renamed,
            on='time',
            direction='backward'
        )
        result = df_combined.dropna().reset_index(drop=True)
        self.logger.info(f"Синхронізовано: {len(result)} рядків M1+M5.")
        return result

    def save_parquet(self, df, suffix=""):
        path = f"data/{self.symbol}_{suffix}.parquet"
        df.to_parquet(path, index=False)
        self.logger.info(f"Збережено: {path}")
        return path