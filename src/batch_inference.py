# src/batch_inference.py

import pandas as pd
import joblib

def batch_predict(model_path, scaler, input_path, output_path):
    model = joblib.load(model_path)
    df = pd.read_csv(input_path)

    X = scaler.transform(df.drop('Outcome', axis=1))  # assuming 'Outcome' is present
    preds = model.predict(X)
    df['prediction'] = preds
    df.to_csv(output_path, index=False)
    print(f"Saved predictions to {output_path}")
