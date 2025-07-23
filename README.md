# diabetes-ml-pipeline

This project demonstrates a complete modular ML pipeline for diabetes prediction using Optuna for hyperparameter tuning and MLflow for experiment tracking.



## Steps to Run
Preprocess Data
Run src/data_preprocessing.py 
Cleans and prepares raw data.
Outputs cleaned datasets for training/testing.


Hyperparameter Tuning
Run src/hpo_optuna.py
Trains a Gradient Boosting Classifier.
Logs model metrics (accuracy, AUC) to MLflow.
Saves model as models/model.pkl.


Train Final Model
Run src/train_model.py 
Uses Optuna to optimize hyperparameters.
Logs all trials to MLflow.


Register Model
Run src/model_registration.py 
Registers the final trained model to MLflow Model Registry.
Saves final model to models/final_model.pkl.


Batch Inference
Run src/batch_inference.py 
Loads final model and generates predictions for batch inputs.
Outputs predictions to data/output_predictions.csv.


(Optional)
Data Drift Detection
Run src/drift_detection.py
Performs manual drift detection using KS-test.
Visualizes feature distributions with KDE plots.

## Screenshots

images/1.png

images/2.png

images/2-1.png

images/3.png

images/4.png

images/5.png

images/6.png

images/61.png

images/62.png

images/63.png

images/64.png