# Зв'язок з MetaTrader 5

import logging
import socket

class MT5Connector:
    def __init__(self, config):
        self.host = config['mt5']['host']
        self.port = config['mt5']['port']
        self.logger = logging.getLogger("Bot.Connector")

    def listen_for_data(self):
        """Зчитування ВСІХ даних (циклічний буфер)"""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind((self.host, self.port))
                s.listen(1)
                self.logger.info(f"Очікування даних на {self.host}:{self.port}...")
                conn, addr = s.accept()
                with conn:
                    chunks = []
                    while True:
                        chunk = conn.recv(65536)
                        if not chunk: break
                        chunks.append(chunk)
                    full_data = b"".join(chunks).decode('utf-8', errors='ignore')
                    self.logger.info(f"Отримано пакет: {len(full_data)} байт.")
                    return full_data
            except Exception as e:
                self.logger.error(f"Помилка сокета: {e}")
                return None