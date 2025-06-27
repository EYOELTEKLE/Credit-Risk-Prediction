import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import mlflow
import mlflow.sklearn


def train_model(processed_data_path, target_col='is_high_risk', experiment_name='CreditRiskExperiment'):
    """
    Train models, tune hyperparameters, evaluate, and register best model with MLflow.
    """
    # Load data
    df = pd.read_csv(processed_data_path)
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Define models & hyperparameters
    models = {
        'logreg': (LogisticRegression(solver='liblinear', random_state=42), {
            'C': [0.01, 0.1, 1, 10],
            'penalty': ['l1', 'l2']
        }),
        'rf': (RandomForestClassifier(random_state=42), {
            'n_estimators': [50, 100],
            'max_depth': [3, 5, 10]
        })
    }

    best_score = -np.inf
    best_model = None
    best_model_name = None
    best_run_id = None

    mlflow.set_experiment(experiment_name)

    for name, (model, params) in models.items():
        with mlflow.start_run(run_name=name) as run:
            search = GridSearchCV(model, params, scoring='roc_auc', cv=3, n_jobs=-1)
            search.fit(X_train, y_train)
            y_pred = search.predict(X_test)
            y_proba = search.predict_proba(X_test)[:,1] if hasattr(search, 'predict_proba') else y_pred
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, zero_division=0),
                'recall': recall_score(y_test, y_pred, zero_division=0),
                'f1': f1_score(y_test, y_pred, zero_division=0),
                'roc_auc': roc_auc_score(y_test, y_proba)
            }
            mlflow.log_params(search.best_params_)
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(search.best_estimator_, f"model_{name}")
            print(f"{name} metrics:", metrics)
            # Track best
            if metrics['roc_auc'] > best_score:
                best_score = metrics['roc_auc']
                best_model = search.best_estimator_
                best_model_name = name
                best_run_id = run.info.run_id

    # Register best model
    if best_model is not None:
        with mlflow.start_run(run_id=best_run_id):
            mlflow.sklearn.log_model(best_model, "best_model", registered_model_name="CreditRiskBestModel")
        print(f"Best model ({best_model_name}) registered in MLflow Model Registry.")
    else:
        print("No model was successfully trained.")

# Example usage:
# train_model('data/processed/model_ready_labeled.csv')
