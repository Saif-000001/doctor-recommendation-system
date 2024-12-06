from fastapi import FastAPI, HTTPException
from .models.request_models import SymptomsRequest, DoctorRequest, TreatmentRequest
from .models.response_models import PredictionSymptoms, PredictionDoctor, PredictionTreatment
from .services.prediction import prediction_symptoms, prediction_doctor, prediction_treatment
from .core.config import settings


# Initialize FastAPI app
app = FastAPI(
    title=settings.API_TITLE,
    description=settings.API_DESCRIPTION,
    version=settings.API_VERSION
)

@app.post("/symptom", response_model=PredictionSymptoms)
async def predict_symptoms(request: SymptomsRequest):
    try:
        symptoms = prediction_symptoms.extract_symptoms(request.text)
        return PredictionSymptoms(symptoms=symptoms)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction symptoms section failed: {str(e)}")


@app.post("/doctor", response_model=PredictionDoctor)
async def predict_disease(request: DoctorRequest):
    try:
        symptoms = prediction_symptoms.extract_symptoms(request.text)
        if not symptoms:
            raise HTTPException(
                status_code=400,
                detail="No symptoms could be detected in the provided text"
            )
        
        # Make prediction
        result = prediction_doctor.predict_doctor(symptoms)
        return result
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction doctor section failed: {str(e)}")

@app.post('/treatment', response_model=PredictionTreatment)
async def predict_treatment(request: TreatmentRequest):
    try:
        # Extract symptoms first
        symptoms = prediction_symptoms.extract_symptoms(request.text)
        
        if not symptoms:
            raise HTTPException(
                status_code=400,
                detail="No symptoms could be detected in the provided text"
            )
        
        # Convert symptoms list to comma-separated string
        symptoms_str = ', '.join(symptoms)
        
        # Predict treatment based on symptoms
        treatment = prediction_treatment.predict_tretment(symptoms_str)
        
        # Map the treatment result to the PredictionTreatment model
        return PredictionTreatment(
            predicted_disease=treatment.get("Predicted disease", ""),
            description=treatment.get(" Descriptions ", ""),
            precautions=treatment.get(" Precautions ", []),
            medications=treatment.get(" Medications ", ""),
            workout=treatment.get(" Workout ", []),
            diets=treatment.get(" Diets ", "")
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction treatment section failed: {str(e)}")

@app.get("/symptoms")
async def get_valid_symptoms():
    return {"symptoms": prediction_symptoms.valid_symptoms}

@app.get("/")
async def health_check():
    return {"status": "healthy"}


