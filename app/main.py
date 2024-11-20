from fastapi import FastAPI, HTTPException
from .models.request_models import SymptomRequest
from .models.response_models import PredictionResponse
from .services.prediction import prediction_system
from .core.config import settings
from .utils.text_processing import preprocess_text

# Initialize FastAPI app
app = FastAPI(
    title=settings.API_TITLE,
    description=settings.API_DESCRIPTION,
    version=settings.API_VERSION
)


@app.post("/predict", response_model=PredictionResponse)
async def predict_disease(request: SymptomRequest):
    try:
        # Process text and extract symptoms
        processed_text = preprocess_text(request.text)
        symptoms = prediction_system.extract_symptoms(processed_text)
        print(symptoms)
        if not symptoms:
            raise HTTPException(
                status_code=400,
                detail="No symptoms could be detected in the provided text"
            )
        
        # Make prediction
        result = prediction_system.predict_disease(symptoms)
        return result
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}")

@app.get("/symptoms")
async def get_valid_symptoms():
    return {"symptoms": prediction_system.valid_symptoms}

@app.get("/")
async def health_check():
    return {"status": "healthy"}


