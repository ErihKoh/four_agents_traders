import yaml
import logging
from src.connector import MT5Connector
from src.data_loader import DataPipeline
from src.processor import FeatureEngineer


def main():
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Ініціалізація з єдиним джерелом істини (config)
    conn = MT5Connector(config)
    loader = DataPipeline(config)
    processor = FeatureEngineer(config)

    raw_data = conn.listen_for_data()
    if raw_data:
        df_synced = loader.parse_combined_data(raw_data)
        if df_synced is not None:
            df_final = processor.process(df_synced, is_training=True)
            path = loader.save_parquet(df_final, suffix="ml_ready")
            print(f"✅ Готово! Результат у {path}. Таргет:\n{df_final['target'].value_counts()}")


if __name__ == "__main__":
    main()