import yaml
import logging
import joblib
import os
import time
import pandas as pd
from src.connector import MT5Connector
from src.data_loader import DataPipeline
from src.processor import FeatureEngineer
from src.strategy import VirtualAccountant


def setup_logging(config):
    log_dir = config['paths']['log_dir']
    os.makedirs(log_dir, exist_ok=True)

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


def main():
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    setup_logging(config)
    logger = logging.getLogger("Bot.Live")

    try:
        logger.info("Завантаження моделі та ініціалізація...")
        model = joblib.load(f"{config['paths']['models_dir']}/xgb_baseline.pkl")

        conn = MT5Connector(config)
        loader = DataPipeline(config)
        processor = FeatureEngineer(config)
        accountant = VirtualAccountant(config)

        threshold = config['trading'].get('threshold', 0.58)
        logger.info(f"🤖 Бот готовий. Поріг: {threshold} | Баланс: ${accountant.balance:.2f}")
    except Exception as e:
        logger.error(f"Помилка ініціалізації: {e}")
        return

    logger.info("📡 Очікування даних від MT5...")

    while True:
        try:
            # 1. Отримання пакету
            raw_data = conn.listen_for_data()
            if not raw_data:
                time.sleep(1)
                continue

            # 2. Парсинг
            df_raw = loader.parse_combined_data(raw_data)
            if df_raw is None or df_raw.empty:
                time.sleep(1)
                continue

            # Зберігаємо ціну/час ДО обробки процесором
            curr_price = float(df_raw['close'].iloc[-1])
            curr_time = pd.to_datetime(df_raw['time'].iloc[-1])

            # 3. Обробка фіч — тільки останні 100 свічок для інференсу
            df_features = processor.process(df_raw.tail(100), is_training=False)

            if df_features.empty:
                print(f"⏳ Накопичення історії: {len(df_raw)}/100 свічок...", end='\r')
                time.sleep(1)
                continue

            # 4. Перевірка відкритих угод
            accountant.check_pending(curr_time, curr_price)

            # 5. Прогноз на останній свічці
            # 1. Отримуємо список фіч у правильному порядку прямо з "мізків" моделі
            expected_features = model.get_booster().feature_names

            # 2. Беремо останній рядок і вибираємо колонки СУВОРО за списком моделі
            X = df_features[expected_features].tail(1)

            # 3. Тепер predict_proba спрацює без помилок
            prob_up = model.predict_proba(X)[0, 1]

            # Для діагностики можна вивести в лог:
            logging.info(f"📊 Ймовірність UP: {prob_up:.4f}")

            print(
                f"💓 {curr_time.strftime('%H:%M:%S')} | "
                f"Price: {curr_price:.5f} | "
                f"Prob Up: {prob_up:.4f} | "
                f"Balance: ${accountant.balance:.2f}"
            )

            # 6. Логіка входу
            if prob_up > threshold:
                accountant.open_trade(curr_time, curr_price, 'BUY', prob_up)
                logger.info(f"🔥 BUY SIGNAL | Prob: {prob_up:.4f}")
            elif prob_up < (1 - threshold):
                accountant.open_trade(curr_time, curr_price, 'SELL', 1 - prob_up)
                logger.info(f"🔥 SELL SIGNAL | Prob: {1 - prob_up:.4f}")

        except KeyboardInterrupt:
            logger.info("⛔ Бот зупинений користувачем.")
            break
        except Exception as e:
            logger.error(f"Помилка в циклі: {e}", exc_info=True)
            time.sleep(2)


if __name__ == "__main__":
    main()