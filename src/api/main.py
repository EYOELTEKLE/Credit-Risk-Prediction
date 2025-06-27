from fastapi import FastAPI
from .pydantic_models import PredictionRequest, PredictionResponse
import mlflow.pyfunc
import numpy as np

app = FastAPI()

# Load the latest model from MLflow Model Registry at startup
MODEL_NAME = "CreditRiskBestModel"
model = mlflow.pyfunc.load_model(model_uri=f"models:/{MODEL_NAME}/Production")

@app.get("/")
def read_root():
    return {"message": "Credit Risk Prediction API"}

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    # Convert request to DataFrame
    input_df = request.dict()
    input_df = {k: [v] for k, v in input_df.items()}
    input_df = np.array([list(input_df.values())]).reshape(1, -1)
    # Predict risk probability (assume binary classification, get probability of class 1)
    proba = model.predict(input_df)
    if hasattr(proba, '__getitem__') and hasattr(proba[0], '__getitem__'):
        risk_score = float(proba[0][1])
    else:
        risk_score = float(proba[0])
    return PredictionResponse(risk_score=risk_score)
