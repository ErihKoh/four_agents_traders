# Точка входу (Online Engine)

# Оновлений main.py
from src.connector import MT5Connector
from src.data_loader import DataPipeline
from src.processor import FeatureEngineer  # Наш новий модуль
import logging


def main():
    logging.basicConfig(level=logging.INFO)

    # MVP 0: Підключення (Socket Server)
    conn = MT5Connector()

    # MVP 1: Отримання даних
    # (Переконайтеся, що в MQL5 ви змінили 100 на 5000!)
    raw_string = conn.listen_for_data()

    if raw_string:
        loader = DataPipeline(symbol="EURUSD")

        # Парсимо та валідуємо сирі дані
        df_raw = loader.parse_raw_data(raw_string)
        df_clean = loader.validate_data(df_raw)

        if df_clean is not None:
            # MVP 2 & 3: Обробка (Фічі + Лейбли)
            processor = FeatureEngineer(horizon=20)
            df_final = processor.process(df_clean)

            # Збереження фінального результату
            path = loader.save_parquet(df_final, suffix="ml_ready")

            print("\n--- УСПІХ! ---")
            print(f"Файл збережено: {path}")
            print(f"Доступні колонки: {list(df_final.columns)}")
            print(f"Приклад target (0/1):\n{df_final['target'].value_counts()}")


if __name__ == "__main__":
    main()