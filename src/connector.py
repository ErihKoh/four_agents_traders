# Зв'язок з MetaTrader 5

# src/connector.py
import logging
import yaml
import socket


class MT5Connector:
    def __init__(self, config_path="config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        self.host = '127.0.0.1'
        self.port = 5555
        self.logger = logging.getLogger("Bot.Connector")

    def start_server(self):
        """Запуск сервера, який чекає на дані від MT5"""
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # Дозволяємо повторне використання порту
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        try:
            self.server_socket.bind((self.host, self.port))
            self.server_socket.listen(5)
            self.logger.info(f"Python сервер запущено на {self.host}:{self.port}")
            self.logger.info("Очікування підключення від MT5 (Mac)...")

            while True:
                client, addr = self.server_socket.accept()
                data = client.recv(65536)  # Отримуємо великий блок даних
                if data:
                    message = data.decode('utf-8')
                    if message.startswith("TIME"):
                        self.logger.info("Отримано свіжі дані від MT5!")
                        return message  # Повертаємо дані для обробки
                client.close()
        except Exception as e:
            self.logger.error(f"Помилка сервера: {e}")
            return None

    def listen_for_data(self):
        """Очікування даних від MT5 з розширеним буфером для Mac"""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.settimeout(60)  # Чекаємо підключення 1 хвилину
            s.bind(('127.0.0.1', 5555))
            s.listen(1)

            conn, addr = s.accept()
            with conn:
                self.logger.info(f"З'єднання від {addr} прийнято. Очікування даних...")
                # Встановлюємо тайм-аут на саме читання
                conn.settimeout(10)

                # Читаємо відразу великий буфер (5000 свічок)
                data = conn.recv(524288)

                if not data:
                    return None

                return data.decode('utf-8', errors='ignore')