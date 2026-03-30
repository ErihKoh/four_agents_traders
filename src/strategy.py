import numpy as np
import pandas as pd
import logging
import os


class MLStrategy:
    def __init__(self, config):
        # Використовуємо асиметричні пороги з нашого нового config.yaml
        self.buy_threshold = config['trading'].get('threshold_buy', 0.65)
        self.sell_threshold = config['trading'].get('threshold_sell', 0.72)

    def generate_signals(self, df, model):
        # 🚀 Отримуємо правильний порядок колонок прямо з моделі
        expected_features = model.get_booster().feature_names

        # Вибираємо колонки саме в тому порядку, який хоче модель
        X = df[expected_features]

        probs = model.predict_proba(X)[:, 1]

        signals = np.zeros(len(df))
        signals[probs >= self.buy_threshold] = 1
        signals[probs <= (1 - self.sell_threshold)] = -1

        return signals, probs


class VirtualAccountant:
    def __init__(self, config):
        self.config = config
        self.balance = float(config['backtest'].get('initial_balance', 1000.0))
        self.risk_pct = float(config['trading'].get('risk_per_trade', 0.01))

        self.spread = float(config['backtest'].get('spread', 0.00012))
        self.sl_points = float(config['backtest'].get('sl_points', 100))
        self.tp_points = float(config['backtest'].get('tp_points', 80))

        # Горизонт у барах (автоматично з нашого нового processor.py)
        target_min = config['trading'].get('target_minutes', 20)
        tf_min = config['trading'].get('timeframe_minutes', 1)
        self.horizon_bars = max(1, target_min // tf_min)

        self.pending_trades = []
        self.logger = logging.getLogger("Signals")

    import pandas as pd

    def open_trade(self, current_time, price, direction, prob, current_vol=None):
        if len(self.pending_trades) > 0:
            return

        # 🚀 МАТЕМАТИКА ДИНАМІЧНОГО ЛОТУ
        # Впевненість моделі (наскільки prob відхиляється від 0.5)
        confidence = abs(prob - 0.5) * 2  # Результат від 0.0 до 1.0

        # Формула: Базовий_ризик * (Множник впевненості)
        base_risk = float(self.config['trading'].get('risk_per_trade', 0.01))
        dynamic_risk = base_risk * (0.5 + confidence)

        # Обмежуємо зверху (макс 2%), щоб не ризикувати надмірно
        dynamic_risk = min(dynamic_risk, base_risk * 2)

        # Отримуємо час витримки з конфігу (наприклад, 120 хвилин)
        target_min = self.config['trading'].get('target_minutes', 60)

        trade = {
            'open_price': price,
            'direction': direction,
            # 🚀 ВИПРАВЛЕННЯ: Додаємо хвилини до Timestamp через Timedelta
            'close_time': current_time + pd.Timedelta(minutes=target_min),
            'risk_pct': dynamic_risk,  # Зберігаємо цей ризик для закриття!
            'is_be': False,
            'trail_pips': None  # Для майбутнього трейлінгу
        }

        self.pending_trades.append(trade)
        # print(f"📡 Відкрито {direction} з ризиком {dynamic_risk*100:.2f}% до {trade['close_time']}")

    def check_pending(self, current_time, current_price):
        """
        Перевіряє активні угоди на вихід по TP, SL, Trailing або TIME.
        Тепер використовує current_time (Timestamp) для контролю часу.
        """
        for trade in self.pending_trades[:]:
            # 1. Розрахунок поточного результату в піпсах
            if trade['direction'] == 'BUY':
                pips = (current_price - trade['open_price']) * 100000
            else:
                pips = (trade['open_price'] - current_price) * 100000

            # 🛡️ ЛОГІКА БЕЗУБИТКУ (BE)
            be_pct = self.config['trading'].get('be_threshold_pct', 0.5)
            if not trade.get('is_be', False) and pips >= (self.tp_points * be_pct):
                trade['is_be'] = True
                # Встановлюємо початковий рівень захисту на +2 піпси (комісія)
                if not trade.get('trail_pips'):
                    trade['trail_pips'] = 2.0

            # 📈 ЛОГІКА TRAILING STOP
            # Активуємо після 60% шляху до цілі
            trail_activation = self.tp_points * 0.6
            if pips >= trail_activation:
                # Трейлінг тримає стоп на відстані 40 піпсів від піку прибутку
                new_trail = pips - 40.0
                if not trade.get('trail_pips') or new_trail > trade['trail_pips']:
                    trade['trail_pips'] = new_trail

            # ⚖️ ВИЗНАЧЕННЯ ПОТОЧНОГО ЛІМІТУ СТОПА
            if trade.get('trail_pips') is not None:
                sl_limit = trade['trail_pips']
            elif trade.get('is_be', False):
                sl_limit = 2.0
            else:
                sl_limit = -self.sl_points

            # 🚨 УМОВИ ЗАКРИТТЯ
            is_sl = pips <= sl_limit
            is_tp = pips >= self.tp_points

            # 🚀 ВИПРАВЛЕННЯ ПОМИЛКИ ЧАСУ: Порівнюємо Timestamp з Timestamp
            is_timeout = current_time >= trade['close_time']

            if is_sl or is_tp or is_timeout:
                # Визначаємо причину для гарного логування
                if is_sl:
                    if trade.get('trail_pips') and trade['trail_pips'] > 2.0:
                        reason = "TRAIL"
                    elif trade.get('is_be'):
                        reason = "BE"
                    else:
                        reason = "SL"
                elif is_tp:
                    reason = "TP"
                else:
                    reason = "TIME"

                self._close_trade(trade, pips, reason)

    def _close_trade(self, trade, pips, reason):
        # 1. Враховуємо спред (віднімаємо його від результату в піпсах)
        spread_pips = self.spread * 100000
        net_pips = pips - spread_pips

        # 2. ДИНАМІЧНИЙ РИЗИК
        # Беремо ризик, який був розрахований саме для ЦІЄЇ угоди в open_trade
        # Якщо його немає (стара версія), беремо дефолтний з конфігу
        current_risk_pct = trade.get('risk_pct', self.risk_pct)

        # 3. РОЗРАХУНОК ПРИБУТКУ / ЗБИТКУ
        # Формула: (Отримані піпси / Плановий стоп) * (Сума ризику в $)
        # Це гарантує, що при закритті по SL ми втратимо рівно risk_pct
        risk_amount = self.balance * current_risk_pct
        profit = (net_pips / self.sl_points) * risk_amount

        # 4. ОНОВЛЕННЯ БАЛАНСУ
        self.balance += profit

        # Видаляємо угоду зі списку активних
        if trade in self.pending_trades:
            self.pending_trades.remove(trade)

        # 5. ГАРНИЙ ВИВІД У КОНСОЛЬ
        # Додаємо емодзі для різних типів закриття
        if reason == "TP":
            status = "🎯 TP"
        elif reason == "TRAIL":
            status = "📈 TRAIL"
        elif reason == "BE":
            status = "🛡 BE"
        elif profit > 0:
            status = "✅ WIN"
        else:
            status = "❌ LOSS"

        print(
            f"{status} ({reason:5}) | Pips: {net_pips:6.1f} | Risk: {current_risk_pct * 100:.2f}% | Bal: ${self.balance:.2f}")