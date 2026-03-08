import pandas as pd
import numpy as np


class VectorizedBacktester:
    def __init__(self, config):
        self.spread = config['backtest'].get('spread', 0.00010)
        self.horizon = config['trading']['horizon']

    def run(self, df, signals):
        res = df[['time', 'close']].copy()
        res['signal'] = signals

        # Рух ціни через горизонт
        res['price_change'] = df['close'].shift(-self.horizon) - df['close']

        # Прибуток = (Сигнал * Зміна) - Спред (тільки якщо був сигнал)
        res['raw_profit'] = res['signal'] * res['price_change']
        res['costs'] = np.where(res['signal'] != 0, self.spread, 0)
        res['net_profit'] = res['raw_profit'] - res['costs']

        res['equity'] = res['net_profit'].cumsum()
        return self._calculate_metrics(res)

    def _calculate_metrics(self, res):
        net_profit = res['net_profit'].sum()
        trades = res[res['signal'] != 0]
        num_trades = len(trades)

        win_rate = len(res[res['net_profit'] > 0]) / num_trades if num_trades > 0 else 0

        # Розрахунок Profit Factor
        wins = res[res['net_profit'] > 0]['net_profit'].sum()
        losses = abs(res[res['net_profit'] < 0]['net_profit'].sum())
        pf = wins / losses if losses > 0 else 0

        return {
            "Net Profit (pts)": round(net_profit * 100000, 1),
            "Trades Count": num_trades,
            "Win Rate (%)": round(win_rate * 100, 1),
            "Profit Factor": round(pf, 2)
        }, res