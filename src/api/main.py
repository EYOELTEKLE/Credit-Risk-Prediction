from fastapi import FastAPI
from .pydantic_models import PredictionRequest, PredictionResponse

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Credit Risk Prediction API"}

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    # Dummy prediction logic
    return PredictionResponse(risk_score=0.5)
