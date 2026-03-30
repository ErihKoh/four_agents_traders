import pandas as pd
import xgboost as xgb
import joblib
import os
import logging
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report
)
from sklearn.model_selection import TimeSeriesSplit
from src.processor import FeatureEngineer


class ModelTrainer:
    def __init__(self, config):
        self.config = config
        self.model_cfg = config['model']
        self.models_dir = config['paths']['models_dir']
        self.processor = FeatureEngineer(config)
        self.logger = logging.getLogger("Bot.Trainer")

    def train(self, df):
        self.logger.info(f"🚀 Навчання {self.model_cfg['type']} моделі...")

        df = self.processor.process(df, is_training=True)

        if 'target' not in df.columns or df.empty:
            self.logger.error("❌ Колонка 'target' не створена або df порожній.")
            return {"Accuracy": 0}

        X = df.drop(columns=['time', 'target'], errors='ignore')
        y = df['target']

        # Walk-Forward валідація
        self.logger.info("📊 Walk-Forward Cross-Validation (5 фолдів)...")
        tscv = TimeSeriesSplit(n_splits=5)
        cv_scores = []
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
            m = xgb.XGBClassifier(
                n_estimators=self.model_cfg['n_estimators'],
                max_depth=self.model_cfg['max_depth'],
                learning_rate=self.model_cfg['learning_rate'],
                subsample=0.8,
                colsample_bytree=0.8,
                eval_metric='logloss',
                random_state=42
            )
            m.fit(X.iloc[train_idx], y.iloc[train_idx], verbose=False)
            score = accuracy_score(y.iloc[val_idx], m.predict(X.iloc[val_idx]))
            cv_scores.append(score)
            self.logger.info(f"  Фолд {fold}: Accuracy={score:.4f}")

        self.logger.info(
            f"CV Accuracy: {sum(cv_scores)/len(cv_scores):.4f} "
            f"± {pd.Series(cv_scores).std():.4f}"
        )

        # Фінальна модель на train частині з early stopping
        split_idx = int(len(df) * (1 - self.model_cfg['test_size']))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        model = xgb.XGBClassifier(
            n_estimators=500,  # більше дерев — early stopping зупинить вчасно
            max_depth=self.model_cfg['max_depth'],
            learning_rate=self.model_cfg['learning_rate'],
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric='logloss',
            early_stopping_rounds=50,
            random_state=42
        )
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False
        )
        self.logger.info(f"Best iteration: {model.best_iteration}")

        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)[:, 1]

        metrics = {
            "Accuracy":  accuracy_score(y_test, preds),
            "Precision": precision_score(y_test, preds, zero_division=0),
            "Recall":    recall_score(y_test, preds, zero_division=0),
            "F1-Score":  f1_score(y_test, preds, zero_division=0),
            "ROC-AUC":   roc_auc_score(y_test, probs)
        }

        self._display_report(y_test, preds, metrics, model, X.columns)

        os.makedirs(self.models_dir, exist_ok=True)
        joblib.dump(model, os.path.join(self.models_dir, "xgb_baseline.pkl"))
        self.logger.info("✅ Модель збережена.")

        return metrics

    def _display_report(self, y_true, y_pred, metrics, model, feature_names):
        print("\n" + "=" * 55)
        print(f"📊 ЗВІТ МОДЕЛІ ({self.model_cfg['type'].upper()})")
        print("=" * 55)
        for name, value in metrics.items():
            print(f"  {name:12}: {value:.4f}")
        print("-" * 55)
        print(classification_report(y_true, y_pred, target_names=['Down', 'Up']))
        print("-" * 55)
        importances = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        print("🏆 ТОП-7 ФІЧ:")
        print(importances.head(7).to_string(index=False))
        print("=" * 55)