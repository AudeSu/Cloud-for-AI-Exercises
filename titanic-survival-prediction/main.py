from src.data_processing import load_and_preprocess_data
from src.model_training import train_model
from src.prediction import evaluate_model, print_metrics

def main():
    filepath = "data/Titanic-Dataset.csv"
    X_train, X_test, y_train, y_test = load_and_preprocess_data(filepath)
    model = train_model(X_train, y_train)
    metrics = evaluate_model(model, X_test, y_test)
    print_metrics(metrics)

if __name__ == "__main__":
    main()
