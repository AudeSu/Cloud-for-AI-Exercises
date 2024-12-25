# Exercise 2
# from src.data_processing import load_and_preprocess_data
# from src.model_training import train_model
# from src.prediction import evaluate_model, print_metrics

# def main():
#     filepath = "data/Titanic-Dataset.csv"
#     X_train, X_test, y_train, y_test = load_and_preprocess_data(filepath)
#     model = train_model(X_train, y_train)
#     metrics = evaluate_model(model, X_test, y_test)
#     print_metrics(metrics)

# if __name__ == "__main__":
#     main()

import pickle
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd

app = FastAPI(title="Titanic Survival Prediction API")

# Load the trained model and scaler
with open('models/model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

class TitanicPassenger(BaseModel):
    Pclass: int
    Sex: int  # 0 for male, 1 for female
    Age: float
    SibSp: int
    Parch: int
    Fare: float
    Embarked_Q: int  # 0 or 1
    Embarked_S: int  # 0 or 1

class PredictionResponse(BaseModel):
    survival_prediction: int
    survival_probability: float

@app.get("/")
def read_root():
    return {"message": "Welcome to the Titanic Survival Prediction API"}

@app.post("/predict", response_model=PredictionResponse)
def predict_survival(passenger: TitanicPassenger):
    # Convert input data to numpy array
    input_data = np.array([[
        passenger.Pclass,
        passenger.Sex,
        passenger.Age,
        passenger.SibSp,
        passenger.Parch,
        passenger.Fare,
        passenger.Embarked_Q,
        passenger.Embarked_S
    ]])
    
    # Scale the input data
    scaled_data = scaler.transform(input_data)
    
    # Make prediction
    prediction = model.predict(scaled_data)[0]
    probability = model.predict_proba(scaled_data)[0][1]
    
    return PredictionResponse(
        survival_prediction=int(prediction),
        survival_probability=float(probability)
    )