import numpy as np
import pandas as pd
import logging


class MLStrategy:
    def __init__(self, config):
        self.threshold = config['trading'].get('threshold', 0.58)

    def generate_signals(self, df, model):
        # Видаляємо все зайве, що не є фічами для XGBoost
        X = df.drop(columns=['time', 'target'], errors='ignore')

        # Отримуємо ймовірність росту (клас 1)
        probs = model.predict_proba(X)[:, 1]

        # Сигнали: 1 (BUY), -1 (SELL), 0 (WAIT)
        signals = [1 if p > self.threshold else (-1 if p < (1 - self.threshold) else 0) for p in probs]
        return np.array(signals), probs  # Повертаємо і сигнали, і ймовірності для логів


class VirtualAccountant:
    def __init__(self, config):
        self.config = config
        self.balance = float(config['backtest'].get('initial_balance', 1000.0))
        self.risk_pct = float(config['trading'].get('risk_per_trade', 0.01))
        self.spread = float(config['backtest'].get('spread', 0.00012))
        self.sl_points = float(config['backtest'].get('sl_points', 150))  # Напр. 150 пункти
        self.horizon = int(config['trading'].get('horizon', 20))

        self.pending_trades = []
        self.logger = logging.getLogger("Signals")

    def open_trade(self, time, price, direction, prob):
        # Заборона відкривати однакові угоди, поки стара не закрита
        if any(t['direction'] == direction for t in self.pending_trades):
            return

        new_trade = {
            'open_time': time,
            'open_price': price,
            'direction': direction,
            'prob': prob,
            'close_time': time + pd.Timedelta(minutes=self.horizon)
        }
        self.pending_trades.append(new_trade)

        self.logger.info(f"🚀 OPEN {direction} | Price: {price:.5f} | Prob: {prob:.2f} | SL: {self.sl_points} pts")

    def check_pending(self, current_time, current_price):
        for trade in self.pending_trades[:]:
            if trade['direction'] == 'BUY':
                diff_points = (current_price - trade['open_price']) * 100000
            else:
                diff_points = (trade['open_price'] - current_price) * 100000

            is_time_up = current_time >= trade['close_time']
            is_sl_hit = diff_points <= -self.sl_points

            if is_time_up or is_sl_hit:
                reason = "TIMEOUT" if is_time_up else "STOP-LOSS"
                self._close_trade(trade, current_price, diff_points, current_time, reason)

    def _close_trade(self, trade, close_price, pips, close_time, reason):
        # 1. Враховуємо спред (віднімаємо його від результату в пунктах)
        spread_points = self.spread * 100000
        net_pips = pips - spread_points

        # 2. НОВА ФОРМУЛА ПРИБУТКУ:
        # Логіка: Якщо ми втрачаємо sl_points, ми втрачаємо risk_amount.
        # Формула: Profit = (NetPips / SL_Points) * RiskAmount
        risk_amount = self.balance * self.risk_pct
        profit_usd = (net_pips / self.sl_points) * risk_amount

        self.balance += profit_usd
        self.pending_trades.remove(trade)

        status = "✅ WIN" if profit_usd > 0 else "❌ LOSS"
        msg = f"{status} ({reason}) | Pips: {net_pips:.1f} | Profit: ${profit_usd:.2f} | Bal: ${self.balance:.2f}"

        self.logger.info(msg)
        print(f"💰 {msg}")