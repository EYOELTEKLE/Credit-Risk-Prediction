from pydantic import BaseModel

class PredictionRequest(BaseModel):
    # Example features, update with actual model features as needed
    total_amount: float
    avg_amount: float
    transaction_count: float
    std_amount: float
    transaction_hour: float
    transaction_day: float
    transaction_month: float
    transaction_year: float
    # Add more features as required by your actual model

class PredictionResponse(BaseModel):
    risk_score: float
