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
    """Налаштування логування: тех-логи в файл/консоль, сигнали - тільки в файл."""
    log_dir = config['paths']['log_dir']
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # 1. Загальний логгер (Root)
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers = []  # Очищення від старих хендлерів

    # Файл для технічних логів
    fh = logging.FileHandler(f"{log_dir}/bot_core.log")
    fh.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(message)s'))
    root_logger.addHandler(fh)

    # Консоль (чистий вивід)
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
    root_logger.addHandler(ch)

    # 2. Логгер сигналів (Окремий файл без дублювання в консоль)
    signal_logger = logging.getLogger("Signals")
    signal_logger.propagate = False  # Не пускати лог в консоль двічі
    sig_handler = logging.FileHandler(f"{log_dir}/signals.log")
    sig_handler.setFormatter(logging.Formatter('%(asctime)s | %(message)s'))
    signal_logger.addHandler(sig_handler)
    signal_logger.setLevel(logging.INFO)

    return signal_logger


def main():
    # 1. Завантаження конфігурації
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # 2. Налаштування логів
    sig_log = setup_logging(config)
    logger = logging.getLogger("Bot.Live")

    # 3. Ініціалізація компонентів
    try:
        logger.info("Завантаження моделі XGBoost та компонентів...")
        model = joblib.load(f"{config['paths']['models_dir']}/xgb_baseline.pkl")

        conn = MT5Connector(config)
        loader = DataPipeline(config)
        processor = FeatureEngineer(config)
        accountant = VirtualAccountant(config)

        logger.info(f"🤖 Бот готовий. Баланс: ${accountant.balance}. Поріг: {config['trading']['threshold']}")
    except Exception as e:
        logger.error(f"Критична помилка ініціалізації: {e}")
        return

    # 4. Нескінченний цикл Live-торгівлі
    logger.info("📡 Очікування пакетів даних від MetaTrader 5...")
    while True:
        try:
            # Чекаємо дані від MT5
            raw_data = conn.listen_for_data()
            if not raw_data:
                continue

            # Парсинг та синхронізація (M1 + M5)
            df_synced = loader.parse_combined_data(raw_data)

            if df_synced is not None:
                # Обробка ознак (is_training=False використовує існуючий scaler.pkl)
                df_features = processor.process(df_synced, is_training=False)

                # Дані останньої свічки
                last_row = df_features.tail(1)
                curr_time = last_row['time'].iloc[0]
                curr_price = last_row['close'].iloc[0]

                # Крок А: Перевірка та закриття старих угод (через 20 хв)
                accountant.check_pending(curr_time, curr_price)

                # Крок Б: Прогноз для поточної ситуації
                X = last_row.drop(columns=['time'])  # Таргету в Live немає
                prob_up = model.predict_proba(X)[0, 1]
                threshold = config['trading']['threshold']

                # Крок В: Прийняття рішення про вхід
                if prob_up > threshold:
                    accountant.open_trade(curr_time, curr_price, 'BUY', prob_up)
                elif prob_up < (1 - threshold):
                    accountant.open_trade(curr_time, curr_price, 'SELL', 1 - prob_up)
                else:
                    # Просто логуємо стан очікування (опціонально)
                    logger.debug(f"WAIT | Prob: {prob_up:.2f} | Time: {curr_time}")

        except KeyboardInterrupt:
            logger.info("Зупинка бота користувачем. До зустрічі!")
            break
        except Exception as e:
            logger.error(f"Помилка в основному циклі: {e}")


if __name__ == "__main__":
    main()