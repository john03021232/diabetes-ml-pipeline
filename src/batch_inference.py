# src/batch_inference.py
# To generate predictions

import pandas as pd
import joblib
import os

def batch_predict(model_path, scaler, input_path, output_path):
    # Load model
    model = joblib.load(model_path)

    # Load data
    df = pd.read_csv(input_path)
    #debug
    #print('Columns in CSV: ' ,df.columns.tolist())

    # Apply scaler to all columns except target (to match training)
    feature_cols = [col for col in df.columns if col not in ['Diabetic']]  # Only exclude target

    # Apply scaler
    X = scaler.transform(df[feature_cols])


    # Predict classes
    preds = model.predict(X)
    preds_prob = model.predict_proba(X)[:, 1]


    # Add predictions to dataframe
    df['prediction'] = preds
    df['prediction_prob'] = preds_prob

    # Create output directory if needed
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save predictions
    df.to_csv(output_path, index=False)
    print(f"Saved predictions to {output_path}")