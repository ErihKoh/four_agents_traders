import pandas as pd
import numpy as np


class VectorizedBacktester:
    def __init__(self, config):
        self.spread = config['backtest'].get('spread', 0.00012)
        self.horizon = config['trading']['horizon']
        self.sl_points = config['backtest'].get('sl_points', 150)
        self.tp_points = config['backtest'].get('tp_points', 250)

    def run(self, df, signals):
        res = df[['time', 'close']].copy().reset_index(drop=True)
        res['signal'] = signals

        results = []
        for i, row in res.iterrows():
            if row['signal'] == 0:
                results.append(0.0)
                continue

            direction = row['signal']  # 1=BUY, -1=SELL
            entry = row['close']
            net_profit = 0.0

            # Симулюємо рух по свічках горизонту
            for j in range(i + 1, min(i + self.horizon + 1, len(res))):
                curr = res.at[j, 'close']
                diff_pts = (curr - entry) * 100000 * direction

                if diff_pts <= -self.sl_points:
                    net_profit = (-self.sl_points - self.spread * 100000 * 2) / 100000
                    break
                if self.tp_points > 0 and diff_pts >= self.tp_points:
                    net_profit = (self.tp_points - self.spread * 100000 * 2) / 100000
                    break
            else:
                # Таймаут — закриваємо по поточній ціні
                if i + self.horizon < len(res):
                    final = res.at[i + self.horizon, 'close']
                    diff_pts = (final - entry) * 100000 * direction
                    net_profit = (diff_pts - self.spread * 100000 * 2) / 100000

            results.append(net_profit)

        res['net_profit'] = results
        res['equity'] = res['net_profit'].cumsum()

        return self._calculate_metrics(res), res

    def _calculate_metrics(self, res):
        trades = res[res['signal'] != 0]
        num_trades = len(trades)

        if num_trades == 0:
            return {"Net Profit (pts)": 0, "Trades": 0, "Win Rate (%)": 0, "Profit Factor": 0}

        wins = trades[trades['net_profit'] > 0]['net_profit'].sum()
        losses = abs(trades[trades['net_profit'] < 0]['net_profit'].sum())
        win_rate = len(trades[trades['net_profit'] > 0]) / num_trades
        pf = wins / losses if losses > 0 else float('inf')

        return {
            "Net Profit (pts)": round(res['net_profit'].sum() * 100000, 1),
            "Trades Count":     num_trades,
            "Win Rate (%)":     round(win_rate * 100, 1),
            "Profit Factor":    round(pf, 2),
            "Max Drawdown":     round(
                (res['equity'].cummax() - res['equity']).max() * 100000, 1
            )
        }