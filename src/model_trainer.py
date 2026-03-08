import pandas as pd
import xgboost as xgb
import joblib
import os
import logging
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report
)


class ModelTrainer:
    def __init__(self, config):
        self.config = config
        self.model_cfg = config['model']
        self.models_dir = config['paths']['models_dir']
        self.logger = logging.getLogger("Bot.Trainer")

    def train(self, df):
        self.logger.info(f"🚀 Навчання {self.model_cfg['type']} моделі...")

        X = df.drop(columns=['time', 'target'])
        y = df['target']

        # Використовуємо test_size з конфігу (зазвичай 0.2)
        split_idx = int(len(df) * (1 - self.model_cfg['test_size']))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        # Параметри з config.yaml
        model = xgb.XGBClassifier(
            n_estimators=self.model_cfg['n_estimators'],
            max_depth=self.model_cfg['max_depth'],
            learning_rate=self.model_cfg['learning_rate'],
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric='logloss',
            random_state=42
        )

        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)[:, 1]

        metrics = {
            "Accuracy": accuracy_score(y_test, preds),
            "Precision": precision_score(y_test, preds),
            "Recall": recall_score(y_test, preds),
            "F1-Score": f1_score(y_test, preds),
            "ROC-AUC": roc_auc_score(y_test, probs)
        }

        self._display_advanced_report(y_test, preds, metrics, model, X.columns)

        model_path = os.path.join(self.models_dir, "xgb_baseline.pkl")
        joblib.dump(model, model_path)

        return metrics

    def _display_advanced_report(self, y_true, y_pred, metrics, model, feature_names):
        print("\n" + "=" * 55)
        print(f"📊 ПОВНИЙ ЗВІТ МОДЕЛІ (Конфіг: {self.model_cfg['type']})")
        print("=" * 55)
        for name, value in metrics.items():
            print(f"{name:10}: {value:.4f}")
        print("-" * 55)
        print(classification_report(y_true, y_pred, target_names=['Price_Down', 'Price_Up']))
        print("-" * 55)
        importances = pd.DataFrame({'feature': feature_names, 'importance': model.feature_importances_}).sort_values(
            by='importance', ascending=False)
        print("🏆 ТОП-7 ФІЧ:")
        print(importances.head(7).to_string(index=False))
        print("=" * 55 + "\n")