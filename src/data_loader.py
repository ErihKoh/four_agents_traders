import MetaTrader5 as mt5
import pandas as pd
import logging


class DataPipeline:
    def __init__(self, symbol="EURUSD", timeframe=mt5.TIMEFRAME_M1):
        self.symbol = symbol
        self.timeframe = timeframe
        self.logger = logging.getLogger("Bot.DataPipeline")

    def fetch_history(self, count=50000):
        """Завантаження історичних даних"""
        self.logger.info(f"Fetching {count} bars for {self.symbol}")

        # Отримуємо катирування
        rates = mt5.copy_rates_from_pos(self.symbol, self.timeframe, 0, count)

        if rates is None:
            self.logger.error(f"Failed to get rates: {mt5.last_error()}")
            return None

        # Перетворюємо в DataFrame
        df = pd.DataFrame(rates)

        # Конвертуємо час у зрозумілий формат
        df['time'] = pd.to_datetime(df['time'], unit='s')

        # Залишаємо лише потрібні колонки
        df = df[['time', 'open', 'high', 'low', 'close', 'tick_volume']]
        df.columns = ['time', 'open', 'high', 'low', 'close', 'volume']

        return df

    def validate_data(self, df):
        """Перевірка цілісності даних (MVP 1 Check)"""
        if df is None or df.empty:
            return False, "Dataset is empty"

        # 1. Перевірка на пропуски (NaN)
        if df.isnull().values.any():
            self.logger.warning("Found NaN values. Dropping them.")
            df.dropna(inplace=True)

        # 2. Перевірка на дублікати часових міток
        duplicates = df.duplicated(subset=['time']).sum()
        if duplicates > 0:
            self.logger.warning(f"Found {duplicates} duplicate timestamps.")
            df.drop_duplicates(subset=['time'], inplace=True)

        # 3. Пошук 'дірок' у часі (Gaps)
        # Для M1 різниця між свічками має бути рівно 60 секунд
        time_diff = df['time'].diff().dt.total_seconds()
        expected_diff = 60 if self.timeframe == mt5.TIMEFRAME_M1 else 300

        gaps = time_diff[time_diff > expected_diff]
        if not gaps.empty:
            self.logger.warning(f"Detected {len(gaps)} time gaps in history!")

        return True, df

    def save_to_parquet(self, df, filename):
        """Збереження у форматі Parquet для швидкості"""
        path = f"./data/{filename}.parquet"
        df.to_parquet(path, index=False)
        self.logger.info(f"Dataset saved to {path}")

# Приклад використання в межах MVP 0 фундаменту:
# pipeline = DataPipeline(symbol="EURUSD", timeframe=mt5.TIMEFRAME_M1)
# raw_data = pipeline.fetch_history(50000)
# success, clean_data = pipeline.validate_data(raw_data)
# if success:
#     pipeline.save_to_parquet(clean_data, "eurusd_m1_clean")