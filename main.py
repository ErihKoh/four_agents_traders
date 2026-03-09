import yaml
import logging
import joblib
import os
import pandas as pd
from src.connector import MT5Connector
from src.data_loader import DataPipeline
from src.processor import FeatureEngineer
from src.strategy import MLStrategy, VirtualAccountant


def setup_logging(config):
    """Налаштування логів: тех-логи та сигнали."""
    log_dir = config['paths']['log_dir']
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers = []

    fh = logging.FileHandler(f"{log_dir}/bot_core.log")
    fh.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(message)s'))
    root_logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
    root_logger.addHandler(ch)

    signal_logger = logging.getLogger("Signals")
    signal_logger.propagate = False
    sig_handler = logging.FileHandler(f"{log_dir}/signals.log")
    sig_handler.setFormatter(logging.Formatter('%(asctime)s | %(message)s'))
    signal_logger.addHandler(sig_handler)
    signal_logger.setLevel(logging.INFO)

    return signal_logger


def main():
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    sig_log = setup_logging(config)
    logger = logging.getLogger("Bot.Live")

    try:
        logger.info("Завантаження моделі та ініціалізація...")
        model = joblib.load(f"{config['paths']['models_dir']}/xgb_baseline.pkl")

        conn = MT5Connector(config)
        loader = DataPipeline(config)
        processor = FeatureEngineer(config)
        accountant = VirtualAccountant(config)

        threshold = config['trading'].get('threshold', 0.58)
        logger.info(f"🤖 Бот готовий. Поріг: {threshold} | Баланс: ${accountant.balance}")
    except Exception as e:
        logger.error(f"Помилка ініціалізації: {e}")
        return

    logger.info("📡 Чекаю на дані від MT5...")

    while True:
        try:
            # 1. Отримання пакету даних
            raw_data = conn.listen_for_data()
            if not raw_data:
                continue

            # 2. Парсинг (тут ще є 'close' і 'time')
            df_raw = loader.parse_combined_data(raw_data)

            if df_raw is not None and not df_raw.empty:
                # Зберігаємо ціну/час для стратегії (до того як процесор їх видалить)
                curr_price = float(df_raw['close'].iloc[-1])
                curr_time = pd.to_datetime(df_raw['time'].iloc[-1])

                # 3. Обробка ознак (RSI, EMA, dist_m1_m5)
                df_features = processor.process(df_raw, is_training=False)

                # Перевірка на "холодний старт" (потрібно > 50 свічок для EMA)
                if df_features.empty:
                    print(f"⏳ Накопичення історії: {len(df_raw)}/50 свічок...", end='\r')
                    continue

                # 4. Перевірка відкритих угод (Stop Loss або Time-out)
                accountant.check_pending(curr_time, curr_price)

                # 5. Прогноз
                last_row = df_features.tail(1)
                X = last_row.drop(columns=['time', 'target'], errors='ignore')

                # Ймовірність росту (клас 1)
                prob_up = model.predict_proba(X)[0, 1]

                # --- HEARTBEAT: Вивід стану в консоль для кожної свічки ---
                print(f"💓 {curr_time.strftime('%H:%M:%S')} | Price: {curr_price:.5f} | Prob Up: {prob_up:.4f}")

                # 6. Логіка входу
                if prob_up > threshold:
                    accountant.open_trade(curr_time, curr_price, 'BUY', prob_up)
                    logger.info(f"🔥 BUY SIGNAL! Prob: {prob_up:.4f}")

                elif prob_up < (1 - threshold):
                    accountant.open_trade(curr_time, curr_price, 'SELL', 1 - prob_up)
                    logger.info(f"🔥 SELL SIGNAL! Prob: {1 - prob_up:.4f}")

        except KeyboardInterrupt:
            logger.info("Бот зупинений.")
            break
        except Exception as e:
            logger.error(f"Помилка в циклі: {e}")


if __name__ == "__main__":
    main()