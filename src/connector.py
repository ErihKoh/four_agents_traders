import logging
import socket
import time


class MT5Connector:
    def __init__(self, config):
        self.host = config['mt5']['host']
        self.port = config['mt5']['port']
        self.logger = logging.getLogger("Bot.Connector")

    def listen_for_data(self, timeout=15, retries=3):
        """
        Зчитування даних від MT5 з таймаутом та повторними спробами.
        Після отримання надсилає ACK підтвердження.
        """
        for attempt in range(1, retries + 1):
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                    s.settimeout(timeout)
                    s.bind((self.host, self.port))
                    s.listen(1)
                    self.logger.debug(f"Очікування даних на {self.host}:{self.port}...")

                    conn, addr = s.accept()
                    with conn:
                        conn.settimeout(timeout)
                        chunks = []
                        while True:
                            try:
                                chunk = conn.recv(65536)
                                if not chunk:
                                    break
                                chunks.append(chunk)
                            except socket.timeout:
                                break

                        if not chunks:
                            self.logger.warning(f"Порожній пакет. Спроба {attempt}/{retries}")
                            continue

                        full_data = b"".join(chunks).decode('utf-8', errors='ignore')

                        # Надсилаємо ACK підтвердження
                        try:
                            conn.sendall(b"ACK_OK")
                        except Exception:
                            pass  # ACK не критичний — дані вже отримані

                        self.logger.info(f"Отримано пакет: {len(full_data)} байт.")
                        return full_data

            except socket.timeout:
                self.logger.warning(f"Timeout ({timeout}s). Спроба {attempt}/{retries}")
                time.sleep(1)
            except OSError as e:
                self.logger.error(f"Помилка сокета: {e}. Спроба {attempt}/{retries}")
                time.sleep(2)
            except Exception as e:
                self.logger.error(f"Невідома помилка: {e}")
                time.sleep(1)

        self.logger.error("Всі спроби підключення вичерпано. Повертаємо None.")
        return None