import sys
import os
import pytest
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_processing import load_and_preprocess_data
from src.model_training import train_model
from src.prediction import evaluate_model

def test_pipeline():
    X_train, X_test, y_train, y_test = load_and_preprocess_data("data/Titanic-Dataset.csv")
    model = train_model(X_train, y_train)
    metrics = evaluate_model(model, X_test, y_test)
    assert metrics['accuracy'] > 0.7, "Accuracy should be greater than 70%"
