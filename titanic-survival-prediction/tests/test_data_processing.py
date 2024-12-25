import pytest
import pandas as pd
from src.data_processing import load_data, preprocess_data

def test_load_data():
    df = load_data("data/Titanic-Dataset.csv")
    assert not df.empty

def test_preprocess_data():
    df = load_data("data/Titanic-Dataset.csv")
    X_train, X_test, y_train, y_test = preprocess_data(df)
    assert X_train.shape[0] > 0
    assert y_train.shape[0] > 0
