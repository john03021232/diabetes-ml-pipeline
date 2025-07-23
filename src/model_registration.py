# src/model_registration.py
# To log in MLflow.

import mlflow
import mlflow.sklearn
import joblib
import os
import time
from sklearn.metrics import accuracy_score, roc_auc_score
from mlflow.models.signature import infer_signature

def register_model(model, X_test, y_test, params, save_path="models/final_model.pkl"):
    start_time = time.time()

    # Evaluate metrics
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    train_time = time.time() - start_time

    # Save model locally
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    joblib.dump(model, save_path)
    print(f"Final model saved at: {save_path}")

    # Log everything to MLflow
    mlflow.set_experiment("diabetes-prediction")
    with mlflow.start_run(run_name="Final_Model_Registration"):

        mlflow.log_params(params)
        mlflow.log_metrics({
            "accuracy": acc,
            "auc": auc,
            "training_time": train_time
        })
        mlflow.log_artifact(save_path)  # log .pkl file

        # mlflow.sklearn.log_model(model, name = "model")  # log model for registry view

        # Prepare model signature
        input_example = X_test.iloc[:2] if hasattr(X_test, "iloc") else X_test[:2]
        signature = infer_signature(X_test, model.predict(X_test))

        mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            input_example=input_example,
            signature=signature
        )

        print("Model registered to MLflow")

    return acc, auc
