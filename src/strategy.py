import numpy as np


class MLStrategy:
    def __init__(self, config):
        # Беремо threshold (наприклад, 0.58) з блоку trading:
        self.threshold = config['trading'].get('threshold', 0.55)

    def generate_signals(self, df, model):
        """Перетворює ймовірності у сигнальний ряд [-1, 0, 1]"""
        X = df.drop(columns=['time', 'target'])
        probs = model.predict_proba(X)[:, 1]  # Ймовірність "Up"

        signals = []
        for p in probs:
            if p > self.threshold:
                signals.append(1)  # Buy
            elif p < (1 - self.threshold):
                signals.append(-1)  # Sell
            else:
                signals.append(0)  # Neutral

        return np.array(signals)