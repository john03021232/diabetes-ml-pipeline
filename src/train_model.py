# # src/train_model.py

import os
import time
import joblib
import mlflow
import mlflow.sklearn
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.ensemble import GradientBoostingClassifier

def train(X_train, X_test, y_train, y_test, params, save_path="../models/model.pkl"):
    mlflow.set_experiment("diabetes-prediction")

    try:
        # Ensure the directory exists only once
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        start = time.time()
        model = GradientBoostingClassifier(**params)
        model.fit(X_train, y_train)
        end = time.time()

        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

        mlflow.log_params(params)
        mlflow.log_metrics({"accuracy": acc, "auc": auc, "training_time": end - start})
        mlflow.sklearn.log_model(model, "model")

        joblib.dump(model, save_path)
        print(f"Model saved to {save_path}")

        return model, acc, auc

    except Exception as e:
        print(f"Error during training or saving model: {e}")
        raise
