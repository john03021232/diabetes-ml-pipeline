# src/hpo_optuna.py

import optuna
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
import pandas as pd
from src.data_preprocessing import load_and_preprocess

def objective(trial):
    X_train, X_test, y_train, y_test, _ = load_and_preprocess("data/diabetes.csv")

    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 200),
        "max_depth": trial.suggest_int("max_depth", 2, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),
    }

    model = GradientBoostingClassifier(**params)
    model.fit(X_train, y_train)
    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    return auc

def run_optimization():
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20)
    print("Best trial:", study.best_trial.params)
    return study.best_trial.params
