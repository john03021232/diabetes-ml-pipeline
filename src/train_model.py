# # src/train_model.py
# Using best parameters

import os
import time
import joblib
import mlflow
import mlflow.sklearn
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.ensemble import GradientBoostingClassifier
from mlflow.models.signature import infer_signature

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

        # mlflow.sklearn.log_model(model, name = "model")

        # Infer model signature
        input_example = X_test.iloc[:2] if hasattr(X_test, "iloc") else X_test[:2]
        signature = infer_signature(X_test, model.predict(X_test))

        # Log model to MLflow with name, input_example and signature
        mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            input_example=input_example,
            signature=signature
        )

        joblib.dump(model, save_path)
        print(f"Model saved to {save_path}")

        return model, acc, auc

    except Exception as e:
        print(f"Error during training or saving model: {e}")
        raise
