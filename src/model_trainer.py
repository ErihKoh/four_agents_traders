import pandas as pd
import xgboost as xgb
import joblib
import os
import logging
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report
)
# Додаємо імпорт нашого процесора
from src.processor import FeatureEngineer


class ModelTrainer:
    def __init__(self, config):
        self.config = config
        self.model_cfg = config['model']
        self.models_dir = config['paths']['models_dir']
        # Ініціалізуємо процесор тут
        self.processor = FeatureEngineer(config)
        self.logger = logging.getLogger("Bot.Trainer")

    def train(self, df):
        self.logger.info(f"🚀 Навчання {self.model_cfg['type']} моделі...")

        # --- КЛЮЧОВИЙ МОМЕНТ: Створюємо фічі та таргет перед навчанням ---
        df = self.processor.process(df, is_training=True)

        if 'target' not in df.columns:
            self.logger.error("❌ Помилка: Колонка 'target' не створена. Перевір обсяг даних та горизонт!")
            return {"Accuracy": 0}
        # ---------------------------------------------------------------

        X = df.drop(columns=['time', 'target'], errors='ignore')
        y = df['target']

        # Розподіл на Train/Test (без перемішування для часових рядів)
        split_idx = int(len(df) * (1 - self.model_cfg['test_size']))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

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

        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)

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

        importances = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values(by='importance', ascending=False)

        print("🏆 ТОП-7 ФІЧ:")
        print(importances.head(7).to_string(index=False))
        print("=" * 55 + "\n")