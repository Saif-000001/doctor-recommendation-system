from pydantic import BaseModel

class DoctorRequest(BaseModel):
    text: str

class TreatmentRequest(BaseModel):
    text: str

class SymptomsRequest(BaseModel):
    text: str