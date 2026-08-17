from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

# Initialize FastAPI app
app = FastAPI()

# Load the trained model
model = joblib.load('diabetes_prediction_model.joblib') # Make sure this file is in the same directory as app.py

# Define the input data model for the API
class DiabetesPredictor(BaseModel):
    Pregnancies: int
    Glucose: float
    BloodPressure: float
    SkinThickness: float
    Insulin: float
    BMI: float
    DiabetesPedigreeFunction: float
    Age: int

@app.get("/")
def read_root():
    return {"message": "Welcome to the Diabetes Prediction API!"}

@app.post("/predict/")
def predict_diabetes(data: DiabetesPredictor):
    # Convert input data to pandas DataFrame
    input_df = pd.DataFrame([data.dict()])

    # Make prediction
    prediction = model.predict(input_df)[0]
    prediction_proba = model.predict_proba(input_df)[0].tolist()

    # Map numerical prediction to human-readable string
    diagnosis = "Diabetes" if prediction == 1 else "No Diabetes"

    return {
        "prediction": diagnosis,
        "probability_no_diabetes": prediction_proba[0],
        "probability_diabetes": prediction_proba[1]
    }
