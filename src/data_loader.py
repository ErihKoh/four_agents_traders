import pandas as pd
import logging
import os
from io import StringIO


class DataPipeline:
    def __init__(self, symbol="EURUSD"):
        self.symbol = symbol
        self.logger = logging.getLogger("Bot.DataPipeline")

        # Створюємо папку для даних, якщо її немає
        if not os.path.exists('data'):
            os.makedirs('data')

    def parse_raw_data(self, raw_string):
        """
        Перетворює сирий CSV-текст, отриманий через сокет, у DataFrame.
        """
        try:
            if not raw_string or "ERROR" in raw_string:
                self.logger.error(f"Отримано некоректні дані: {raw_string}")
                return None

            # Читаємо рядок як CSV
            csv_data = StringIO(raw_string)
            df = pd.read_csv(csv_data)

            # Стандартизуємо назви колонок (до нижнього регістру)
            df.columns = [col.lower() for col in df.columns]

            # Конвертація часу
            df['time'] = pd.to_datetime(df['time'])

            # Сортуємо за часом (на випадок, якщо MQL5 видав не по порядку)
            df = df.sort_values('time').reset_index(drop=True)

            return df

        except Exception as e:
            self.logger.error(f"Помилка при парсингу CSV: {e}")
            return None

    def validate_data(self, df):
        """
        MVP 1: Валідація даних на цілісність.
        """
        if df is None or df.empty:
            self.logger.error("Датасет порожній. Валідація неможлива.")
            return None

        initial_len = len(df)

        # 1. Видалення дублікатів за часом
        df = df.drop_duplicates(subset=['time'])

        # 2. Видалення порожніх значень (NaN)
        df = df.dropna()

        # 3. Логування змін
        if len(df) < initial_len:
            self.logger.warning(f"Очищення: видалено {initial_len - len(df)} некоректних рядків.")

        # 4. Перевірка на "дірки" в часі (для M1 - 60 секунд)
        time_diffs = df['time'].diff().dt.total_seconds()
        gaps = time_diffs[time_diffs > 60]  # Для форексу вихідні не враховуємо як помилку
        if not gaps.empty:
            self.logger.info(f"Знайдено {len(gaps)} розривів у часі (це нормально для вихідних або пауз у тиках).")

        self.logger.info(f"Валідація успішна. Отримано свічок: {len(df)}")
        return df

    def sync_timeframes(self, df_m1, df_m5):
        """
        Об'єднання M1 та M5 таймфреймів (якщо ви передаєте обидва через сокет).
        """
        self.logger.info("Синхронізація M1 та M5...")

        # Додаємо суфікси для M5
        df_m5 = df_m5.rename(columns={
            'open': 'open_m5', 'high': 'high_m5',
            'low': 'low_m5', 'close': 'close_m5', 'vol': 'vol_m5'
        })

        # Розумне об'єднання (беремо останню відому M5 свічку для кожної M1)
        df_combined = pd.merge_asof(
            df_m1.sort_values('time'),
            df_m5.sort_values('time'),
            on='time',
            direction='backward'
        )

        return df_combined

    def save_parquet(self, df, suffix=""):
        """
        Збереження у формат Parquet (швидкий та компактний).
        """
        try:
            filename = f"data/{self.symbol.replace('.', '')}_{suffix}.parquet"
            df.to_parquet(filename, index=False)
            self.logger.info(f"Дані успішно збережено у файл: {filename}")
            return filename
        except Exception as e:
            self.logger.error(f"Помилка при збереженні Parquet: {e}")
            return None