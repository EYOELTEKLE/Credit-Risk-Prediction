from pydantic import BaseModel

class PredictionRequest(BaseModel):
    # Define your input features here
    feature1: float
    feature2: float

class PredictionResponse(BaseModel):
    risk_score: float
