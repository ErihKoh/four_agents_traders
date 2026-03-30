import pandas as pd
import xgboost as xgb
import joblib
import os
import logging
from sklearn.metrics import accuracy_score, classification_report
from src.processor import FeatureEngineer


class ModelTrainer:
    def __init__(self, config):
        self.config = config
        self.model_cfg = config['model']
        self.models_dir = config['paths']['models_dir']
        # Використовуємо наш новий процесор з Price Action
        self.processor = FeatureEngineer(config)
        self.logger = logging.getLogger("Bot.Trainer")

    def train(self, df):
        self.logger.info(f"🚀 Навчання XGBoost (MacBook Logic)...")

        # Визначаємо фічі
        features = [c for c in df.columns if c not in ['time', 'target']]
        X = df[features]
        y = df['target']

        # 2. Розбивка на Train/Test (без перемішування!)
        split_idx = int(len(df) * (1 - self.model_cfg.get('test_size', 0.2)))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        # 3. Налаштування моделі (Синхронізовано з MacBook)
        # Прибираємо складний Early Stopping для стабільності на малих таймфреймах
        model = xgb.XGBClassifier(
            n_estimators=self.model_cfg.get('n_estimators', 150),
            max_depth=self.model_cfg.get('max_depth', 4),
            learning_rate=self.model_cfg.get('learning_rate', 0.05),
            subsample=0.8,
            colsample_bytree=0.8,
            objective='binary:logistic',
            random_state=42,
            n_jobs=-1  # Використовуємо всі ядра Windows
        )

        self.logger.info(f"📊 Тренування на {len(X_train)} свічках...")
        model.fit(X_train, y_train)

        # 4. Оцінка результатів
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)

        self._display_report(y_test, preds, acc, model, features)

        # 5. Збереження
        os.makedirs(self.models_dir, exist_ok=True)
        model_path = os.path.join(self.models_dir, "xgb_baseline.pkl")
        joblib.dump(model, model_path)

        # Також зберігаємо список фіч, щоб бектестер не помилився
        joblib.dump(features, os.path.join(self.models_dir, "features.pkl"))

        self.logger.info(f"✅ Модель збережена: {model_path}")
        return {"Accuracy": acc}

    def _display_report(self, y_true, y_pred, acc, model, feature_names):
        print("\n" + "=" * 55)
        print(f"📊 ЗВІТ МОДЕЛІ (MacBook Edition)")
        print("=" * 55)
        print(f"  Accuracy    : {acc:.4f}")
        print("-" * 55)
        print(classification_report(y_true, y_pred, target_names=['Down', 'Up']))
        print("-" * 55)

        # Важливість фіч - тут ми побачимо, чи працює Price Action
        importances = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)

        print("🏆 ТОП-7 ФІЧ (Мають бути RSI/EMA/Patterns):")
        print(importances.head(7).to_string(index=False))
        print("=" * 55)
