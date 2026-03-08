import numpy as np
import pandas as pd
import logging


class MLStrategy:
    def __init__(self, config):
        self.threshold = config['trading'].get('threshold', 0.58)

    def generate_signals(self, df, model):
        X = df.drop(columns=['time', 'target'], errors='ignore')
        probs = model.predict_proba(X)[:, 1]
        signals = [1 if p > self.threshold else (-1 if p < (1 - self.threshold) else 0) for p in probs]
        return np.array(signals)


class VirtualAccountant:
    def __init__(self, config):
        self.config = config
        self.balance = float(config['backtest'].get('initial_balance', 1000.0))
        self.risk_pct = float(config['trading'].get('risk_per_trade', 0.01))
        self.spread = float(config['backtest'].get('spread', 0.00012))
        self.sl_points = float(config['backtest'].get('sl_points', 150))
        self.horizon = int(config['trading'].get('horizon', 20))

        self.pending_trades = []
        self.logger = logging.getLogger("Signals")

    def open_trade(self, time, price, direction, prob):
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

        self.logger.info(f"🚀 OPEN {direction} | Price: {price:.5f} | SL: {self.sl_points} pts")
        print(f"🚀 OPEN {direction} at {price:.5f}")

    def check_pending(self, current_time, current_price):
        """Перевірка умов закриття: Час або Stop Loss."""
        for trade in self.pending_trades[:]:
            # Рахуємо поточний результат у пунктах (1.00001 -> 1 пункт)
            if trade['direction'] == 'BUY':
                diff_points = (current_price - trade['open_price']) * 100000
            else:
                diff_points = (trade['open_price'] - current_price) * 100000

            # Умова 1: Вийшов час (20 хв)
            is_time_up = current_time >= trade['close_time']

            # Умова 2: Спрацював аварійний стоп
            is_sl_hit = diff_points <= -self.sl_points

            if is_time_up or is_sl_hit:
                reason = "TIMEOUT" if is_time_up else "STOP-LOSS"
                self._close_trade(trade, current_price, diff_points, current_time, reason)

    def _close_trade(self, trade, close_price, pips, close_time, reason):
        # Віднімаємо спред при закритті (спрощено)
        net_pips = pips - (self.spread * 100000)

        # Розрахунок прибутку в USD
        risk_amount = self.balance * self.risk_pct
        profit_usd = (net_pips / (self.horizon * 10)) * risk_amount

        self.balance += profit_usd
        self.pending_trades.remove(trade)

        status = "✅ WIN" if profit_usd > 0 else "❌ LOSS"
        msg = f"{status} ({reason}) | Pips: {net_pips:.1f} | Profit: ${profit_usd:.2f} | Bal: ${self.balance:.2f}"

        self.logger.info(msg)
        print(f"💰 {msg}")