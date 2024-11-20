import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pickle
from typing import List, Dict
from collections import Counter
from ..core.config import settings
from ..utils.text_processing import extract_words

class DiseasePredictionSystem:
    def __init__(self):
        """Initialize the disease prediction system"""
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
            self.valid_symptoms = list(self.models['Logistic Regression']["model"].feature_names_in_)
            
        except Exception as e:
            raise Exception(f"Error initializing prediction system: {str(e)}")

    def _load_model(self, model_path):
        """Helper function to load a model"""
        try:
            with open(model_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            raise Exception(f"Error loading model {model_path}: {str(e)}")

    def extract_symptoms(self, text: str) -> List[str]:
        """Extract symptoms from text"""
        try:
            words = extract_words(text)
            matched_symptoms = []
            
            for symptom in self.valid_symptoms:
                symptom_parts = set(symptom.lower().split('_'))
                if symptom_parts.issubset(words):
                    matched_symptoms.append(symptom)
            
            return matched_symptoms
        except Exception as e:
            raise Exception(f"Symptom extraction error: {str(e)}")
    def predict_disease(self, symptoms: List[str]) -> Dict:
        """Make disease prediction using all models"""
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
            raise Exception(f"Prediction error: {str(e)}")
prediction_system = DiseasePredictionSystem()