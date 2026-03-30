import numpy as np
import pandas as pd
import logging
import json
import os


class MLStrategy:
    def __init__(self, config):
        self.threshold = config['trading'].get('threshold', 0.58)

    def generate_signals(self, df, model):
        X = df.drop(columns=['time', 'target'], errors='ignore')
        probs = model.predict_proba(X)[:, 1]
        signals = [
            1 if p > self.threshold
            else (-1 if p < (1 - self.threshold) else 0)
            for p in probs
        ]
        return np.array(signals), probs


class VirtualAccountant:
    STATE_FILE = "logs/state.json"

    def __init__(self, config):
        self.config = config
        self.balance = float(config['backtest'].get('initial_balance', 1000.0))
        self.risk_pct = float(config['trading'].get('risk_per_trade', 0.01))
        self.spread = float(config['backtest'].get('spread', 0.00012))
        self.sl_points = float(config['backtest'].get('sl_points', 150))
        self.tp_points = float(config['backtest'].get('tp_points', 250))
        self.horizon = int(config['trading'].get('horizon', 20))
        self.pending_trades = []
        self.logger = logging.getLogger("Signals")

        self._load_state()

    def open_trade(self, time, price, direction, prob):
        """Відкрити нову угоду (не більше однієї в кожен бік)"""
        if any(t['direction'] == direction for t in self.pending_trades):
            return

        new_trade = {
            'open_time': time.isoformat(),
            'open_price': price,
            'direction': direction,
            'prob': round(prob, 4),
            'close_time': (time + pd.Timedelta(minutes=self.horizon)).isoformat()
        }
        self.pending_trades.append(new_trade)
        self._save_state()

        self.logger.info(
            f"🚀 OPEN {direction} | Price: {price:.5f} | "
            f"Prob: {prob:.2f} | SL: {self.sl_points}pts | TP: {self.tp_points}pts"
        )

    def check_pending(self, current_time, current_price):
        """Перевірка відкритих угод на SL / TP / Timeout"""
        for trade in self.pending_trades[:]:
            open_price = trade['open_price']
            direction = trade['direction']
            close_time = pd.to_datetime(trade['close_time'])

            if direction == 'BUY':
                diff_points = (current_price - open_price) * 100000
            else:
                diff_points = (open_price - current_price) * 100000

            is_sl = diff_points <= -self.sl_points
            is_tp = self.tp_points > 0 and diff_points >= self.tp_points
            is_timeout = current_time >= close_time

            if is_sl or is_tp or is_timeout:
                reason = "SL" if is_sl else ("TP" if is_tp else "TIMEOUT")
                self._close_trade(trade, current_price, diff_points, current_time, reason)

    def _close_trade(self, trade, close_price, pips, close_time, reason):
        # Спред платиться двічі: при відкритті і закритті
        spread_points = self.spread * 100000 * 2
        net_pips = pips - spread_points

        risk_amount = self.balance * self.risk_pct
        profit_usd = (net_pips / self.sl_points) * risk_amount

        self.balance += profit_usd
        self.pending_trades.remove(trade)
        self._save_state()

        status = "✅ WIN" if profit_usd > 0 else "❌ LOSS"
        msg = (
            f"{status} ({reason}) | Pips: {net_pips:.1f} | "
            f"Profit: ${profit_usd:.2f} | Balance: ${self.balance:.2f}"
        )
        self.logger.info(msg)
        print(f"💰 {msg}")

    def _save_state(self):
        """Зберігаємо стан на диск (захист від збоїв)"""
        os.makedirs("logs", exist_ok=True)
        state = {
            'balance': self.balance,
            'pending_trades': self.pending_trades
        }
        with open(self.STATE_FILE, 'w') as f:
            json.dump(state, f, indent=2)

    def _load_state(self):
        """Відновлюємо стан після перезапуску"""
        if os.path.exists(self.STATE_FILE):
            try:
                with open(self.STATE_FILE, 'r') as f:
                    state = json.load(f)
                self.balance = state.get('balance', self.balance)
                self.pending_trades = state.get('pending_trades', [])
                self.logger.info(
                    f"Стан відновлено: balance=${self.balance:.2f}, "
                    f"відкритих угод: {len(self.pending_trades)}"
                )
            except Exception as e:
                self.logger.warning(f"Не вдалося завантажити стан: {e}")