import numpy as np
import pandas as pd
import logging
import json
import os


class MLStrategy:
    def __init__(self, config):
        self.buy_threshold = config['trading'].get('buy_threshold', 0.6)
        self.sell_threshold = config['trading'].get('sell_threshold', 0.4)
        self.min_confidence = config['trading'].get('min_confidence_gap', 0.05)

    def generate_signals(self, df, model):
        X = df.drop(columns=['time', 'target'], errors='ignore')
        probs = model.predict_proba(X)[:, 1]

        signals = []

        for p in probs:
            # Фільтр слабких сигналів
            if abs(p - 0.5) < self.min_confidence:
                signals.append(0)
                continue

            if p >= self.buy_threshold:
                signals.append(1)
            elif p <= self.sell_threshold:
                signals.append(-1)
            else:
                signals.append(0)

        return np.array(signals), probs


class VirtualAccountant:
    STATE_FILE = "logs/state.json"

    def __init__(self, config):
        self.config = config
        self.balance = float(config['backtest'].get('initial_balance', 1000.0))
        self.base_risk = float(config['trading'].get('risk_per_trade', 0.01))

        self.spread = float(config['backtest'].get('spread', 0.00012))
        self.sl_points = float(config['backtest'].get('sl_points', 150))
        self.tp_points = float(config['backtest'].get('tp_points', 250))
        self.horizon = int(config['trading'].get('horizon', 20))

        self.pending_trades = []
        self.logger = logging.getLogger("Signals")

        self._load_state()

    def _dynamic_risk(self, prob):
        """
        Більша впевненість → більший ризик
        """
        confidence = abs(prob - 0.5) * 2  # 0..1
        return self.base_risk * (0.5 + confidence)

    def open_trade(self, time, price, direction, prob):
        # не більше 1 позиції в кожен бік
        if any(t['direction'] == direction for t in self.pending_trades):
            return

        risk_pct = self._dynamic_risk(prob)

        trade = {
            'open_time': time.isoformat(),
            'open_price': price,
            'direction': direction,
            'prob': round(prob, 4),
            'risk_pct': risk_pct,
            'close_time': (time + pd.Timedelta(minutes=self.horizon)).isoformat()
        }

        self.pending_trades.append(trade)
        self._save_state()

        self.logger.info(
            f"🚀 OPEN {direction} | Price: {price:.5f} | "
            f"Prob: {prob:.2f} | Risk: {risk_pct:.3f}"
        )

    def check_pending(self, current_time, current_price):
        for trade in self.pending_trades[:]:
            open_price = trade['open_price']
            direction = trade['direction']
            close_time = pd.to_datetime(trade['close_time'])

            if direction == 'BUY':
                pips = (current_price - open_price) * 100000
            else:
                pips = (open_price - current_price) * 100000

            is_sl = pips <= -self.sl_points
            is_tp = self.tp_points > 0 and pips >= self.tp_points
            is_timeout = current_time >= close_time

            if is_sl or is_tp or is_timeout:
                reason = "SL" if is_sl else ("TP" if is_tp else "TIMEOUT")
                self._close_trade(trade, pips, reason, current_time)

    def _close_trade(self, trade, pips, reason, close_time):
        spread_cost = self.spread * 100000 * 2
        net_pips = pips - spread_cost

        risk_amount = self.balance * trade['risk_pct']
        profit = (net_pips / self.sl_points) * risk_amount

        self.balance += profit
        self.pending_trades.remove(trade)
        self._save_state()

        status = "WIN" if profit > 0 else "LOSS"

        msg = (
            f"{status} ({reason}) | Pips: {net_pips:.1f} | "
            f"Profit: ${profit:.2f} | Balance: ${self.balance:.2f}"
        )

        self.logger.info(msg)
        print(f"💰 {msg}")

    def _save_state(self):
        os.makedirs("logs", exist_ok=True)
        state = {
            'balance': self.balance,
            'pending_trades': self.pending_trades
        }
        with open(self.STATE_FILE, 'w') as f:
            json.dump(state, f, indent=2)

    def _load_state(self):
        if os.path.exists(self.STATE_FILE):
            try:
                with open(self.STATE_FILE, 'r') as f:
                    state = json.load(f)
                self.balance = state.get('balance', self.balance)
                self.pending_trades = state.get('pending_trades', [])
                self.logger.info(
                    f"Loaded state: balance=${self.balance:.2f}, "
                    f"trades={len(self.pending_trades)}"
                )
            except Exception as e:
                self.logger.warning(f"State load error: {e}")