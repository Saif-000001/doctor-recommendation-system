from pydantic import BaseModel
from typing import List

class DiseasePrediction(BaseModel):
    disease: str
    confidence: float
    specialist: str
    description: str

class PredictionDoctor(BaseModel):
    detected_symptoms: List[str]
    predictions: List[DiseasePrediction]
    prediction_summary: dict

class PredictionTreatment(BaseModel):
    predicted_disease: str 
    description: str 
    precautions: List[str]
    medications: str
    workout: List[str]
    diets: str

class PredictionSymptoms(BaseModel):
    symptoms: List[str]