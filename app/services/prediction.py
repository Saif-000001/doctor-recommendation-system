import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pickle
from typing import List, Dict
from collections import Counter
from ..core.config import settings
from ..utils.text_processing import extract_words, preprocess_text

class SymptomsPredictionSystem:
    def __init__(self):
        self.model =self._load_model(settings.SUPPORT_VECTOR_CLASSIFICATION_MODEL)
        self.valid_symptoms = list(self.model.feature_names_in_)
    
    def _load_model(self, model_path):
        try:
            with open(model_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            raise Exception(f"Error loading model {model_path}: {str(e)}")

    def extract_symptoms(self, text: str)-> List[str]:
        try:
            en_convert = preprocess_text(text)
            words = extract_words(en_convert)
            matched_symptoms = []
            
            for symptom in self.valid_symptoms:
                symptom_parts = set(symptom.lower().split('_'))
                if symptom_parts.issubset(words):
                    matched_symptoms.append(symptom)
            
            return matched_symptoms
        except Exception as e:
            raise Exception(f"Symptom extraction error: {str(e)}")


class DoctorPredictionSystem:
    def __init__(self):
        try:
            # Load data
            self.doctor_data = pd.read_csv(
                settings.DOCTOR_DATA_PATH,
                encoding='latin1',
                names=['Disease', 'Specialist']
            )
            self.description_data = pd.read_csv(settings.DESCRIPTION_DATA_PATH)
            
            # Fix mappings
            self.doctor_data['Specialist'] = np.where(
                (self.doctor_data['Disease'] == 'Tuberculosis'),
                'Pulmonologist',
                self.doctor_data['Specialist']
            )
            symptom = SymptomsPredictionSystem()
            self._load_model = symptom._load_model
            
            # Load all models
            self.models = {
                'Logistic Regression': {"model": self._load_model(settings.LOGISTIC_REGRESSION_MODEL)},
                'Decision Tree': {"model": self._load_model(settings.DECISION_TREE_MODEL)},
                'Random Forest': {"model": self._load_model(settings.RANDOM_FOREST_MODEL)},
                'SVM': {"model": self._load_model(settings.SVM_MODEL)},
                'Naive Bayes': {"model": self._load_model(settings.NAIVE_BAYES_MODEL)},
                'KNN': {"model": self._load_model(settings.KNN_MODEL)}
            }
            
            # Setup encoder and symptoms list
            self.label_encoder = LabelEncoder()
            self.label_encoder.fit(self.doctor_data['Disease'])
            # Use features from logistic regression as baseline
            self.valid_symptoms =  symptom.valid_symptoms
            
        except Exception as e:
            raise Exception(f"Error initializing doctor construction system: {str(e)}")

    def predict_doctor(self, symptoms: List[str]):
        try:
            if not symptoms:
                raise ValueError("No symptoms provided")
            
            # Create feature vector
            feature_vector = {col: 1 if col in symptoms else 0 for col in self.valid_symptoms}
            input_df = pd.DataFrame([feature_vector])
            
            # List to store all predictions
            predicted = []
            
            # Get predictions from all models
            for _, values in self.models.items():
                predict_disease = values["model"].predict(input_df)
                predict_disease = self.label_encoder.inverse_transform(predict_disease)
                predicted.extend(predict_disease)
            
            # Calculate disease probabilities
            disease_counts = Counter(predicted)
            percentage_per_disease = {
                disease: (count / len(self.models)) * 100 
                for disease, count in disease_counts.items()
            }
            
            # Create results DataFrame
            result_df = pd.DataFrame({
                "Disease": list(percentage_per_disease.keys()),
                "Chances": list(percentage_per_disease.values())
            })
            
            # Merge with doctor and description data
            result_df = result_df.merge(self.doctor_data, on='Disease', how='left')
            result_df = result_df.merge(self.description_data, on='Disease', how='left')
            
            # Convert to dictionary format for response
            results = []
            for _, row in result_df.iterrows():
                # Replace NaN descriptions with a default string
                description = row['Description'] if pd.notna(row['Description']) else "Description not available"
                results.append({
                    "disease": row['Disease'],
                    "confidence": float(row['Chances']),
                    "specialist": row['Specialist'],
                    "description": description
                })
            
            # Sort by confidence in descending order
            results.sort(key=lambda x: x['confidence'], reverse=True)
            
            return {
                "detected_symptoms": symptoms,
                "predictions": results,
                "prediction_summary": {
                    "total_models": len(self.models),
                    "diseases_predicted": len(results),
                    "most_likely_disease": results[0]['disease'] if results else None,
                    "highest_confidence": results[0]['confidence'] if results else 0
                }
            }
            
        except Exception as e:
            raise Exception(f"Doctor Prediction error: {str(e)}")

class TreatmentPredictionSystem:
    def __init__(self):
        try:
            # data 
            self.descriptions_df = pd.read_csv(settings.DESCRIPTION_DATA_PATH)
            self.precautions_df = pd.read_csv(settings.PRECAUTIONS_DATA_PATH)
            self.medications_df = pd.read_csv(settings.MEDICATIONS_DATA_PATH)
            self.diets_df = pd.read_csv(settings.DIETS_DATA_PATH)
            self.workout_df = pd.read_csv(settings.WORKOUT_DATA_PATH)

            # model
            load_model = SymptomsPredictionSystem()
            self.model = load_model._load_model(settings.SUPPORT_VECTOR_CLASSIFICATION_MODEL)
            # Create symptoms dictionary

            self.symptoms_dict = settings.SYMPTOM_DICT
            # Create diseases dictionary  
            self.diseases_list = settings.DISEASES_LIST
        
        except Exception as e:
            raise Exception(f"Error initializing treatment construction system: {str(e)}")
  
    def treatmentFunction(self, predicted_disease):
        try:
            descriptions = self.descriptions_df[self.descriptions_df['Disease'] == predicted_disease]['Description']
            descriptions = " ".join([description for description in descriptions])

            precautions = self.precautions_df[self.precautions_df['Disease'] == predicted_disease][['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']]
            precautions = [precaution for precaution in precautions.values]

            medications = self.medications_df[self.medications_df['Disease'] == predicted_disease]['Medication']
            medications = [med for med in medications.values]

            diets = self.diets_df[self.diets_df['Disease'] == predicted_disease]['Diet']
            diets = [diet for diet in diets.values]

            workout = self.workout_df[self.workout_df['disease'] == predicted_disease] ['workout']
            return descriptions,precautions,medications,diets,workout
        except Exception as e:
            raise Exception(f"Error treatment function system: {str(e)}")

    # Model Prediction function
    def get_predicted_value(self,patient_symptoms):
        try:
            input_vector = np.zeros(len(self.symptoms_dict))
            for item in patient_symptoms:
                input_vector[self.symptoms_dict[item]] = 1
            return self.diseases_list[self.model.predict([input_vector])[0]]
        except Exception as e:
            raise Exception(f"Error get predicted value system: {str(e)}")
    
    def predict_tretment(self, symptoms:str):
        try:
            user_symptoms = [s.strip() for s in symptoms.split(',')]
            user_symptoms = [symptom.strip("[]' ") for symptom in user_symptoms]
            predicted_disease = self.get_predicted_value(user_symptoms)
            descriptions, precautions, medications, diets, workouts = self.treatmentFunction(predicted_disease)

            precautions_result = []
            for i, precaution in enumerate(precautions[0]):
                precautions_result.append(f"{i}: {precaution}")

            workouts_result = []
            for i, workout in enumerate(workouts):
                workouts_result.append(f"{i}: {workout}")

            return {
                "Predicted disease": predicted_disease,
                " Descriptions ": descriptions,
                " Precautions ": precautions_result,
                " Medications ": medications[0] if medications else "",
                " Workout ": workouts_result,
                " Diets ": diets[0] if diets else "",
            }
        except Exception as e:
            raise Exception(f"Error treatment predicted value system: {str(e)}")

prediction_symptoms = SymptomsPredictionSystem()
prediction_doctor = DoctorPredictionSystem()
prediction_treatment = TreatmentPredictionSystem()

# symptom = prediction_symptoms.extract_symptoms('সকাল থেকে পেটে ব্যথা করতেছে, বিকেলে মাথাব্যথা করছিল, রাতে বমি হয়েছে ।')
# doctor = prediction_doctor.predict_doctor(symptom)
# treatment = prediction_treatment.predict_tretment('joint_pain')
# print(symptom)
# print(doctor)
# print(treatment)