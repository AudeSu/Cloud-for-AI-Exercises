def titanic_pipeline(filepath: str):
    """Main pipeline to process data, train models, and evaluate them."""
    # Step 1: Load and preprocess data
    X_train, X_test, y_train, y_test = load_and_preprocess_data(filepath)
    
    # Normalize numeric features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Step 2: Train models
    logreg, best_rf = train_model(X_train_scaled, y_train)
    
    # Step 3: Predict and evaluate
    models = {"Logistic Regression": logreg, "Random Forest": best_rf}
    predict_and_evaluate(models, X_test_scaled, y_test)

# Run the pipeline
if __name__ == "__main__":
    filepath = "Titanic-Dataset.csv"  # Replace with the actual dataset path
    titanic_pipeline(filepath)