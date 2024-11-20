from pydantic import BaseModel
from typing import List

class DiseasePrediction(BaseModel):
    disease: str
    confidence: float
    specialist: str
    description: str

class PredictionResponse(BaseModel):
    detected_symptoms: List[str]
    predictions: List[DiseasePrediction]
    prediction_summary: dict
