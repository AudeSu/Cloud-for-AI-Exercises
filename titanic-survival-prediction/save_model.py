import pickle
from src.data_processing import load_and_preprocess_data
from src.model_training import train_model
import os

def save_trained_model():
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Load and preprocess data
    filepath = "data/Titanic-Dataset.csv"
    X_train, X_test, y_train, y_test, scaler = load_and_preprocess_data(filepath)
    
    # Train model
    model = train_model(X_train, y_train)
    
    # Save model
    with open('models/model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    # Save scaler
    with open('models/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    print("Model and scaler saved successfully in the 'models' directory!")

if __name__ == "__main__":
    save_trained_model()