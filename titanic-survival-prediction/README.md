# Titanic ML Pipeline

This project builds a production-ready ML pipeline for the Titanic survival prediction challenge.

## Structure
- `src/`: Core modules for data processing, model training, and prediction.
- `tests/`: Unit tests for the pipeline.
- `data/`: Dataset storage.
- `notebooks/`: Jupyter notebooks for EDA.

## Usage
1. Install dependencies: `pip install -r requirements.txt`
2. Run tests: `pytest tests/`
3. Execute the pipeline:
   ```python
   from src.data_processing import load_data, preprocess_data, scale_data
   from src.model_training import train_random_forest
   from src.predictions import predict, evaluate_model

   df = load_data("data/Titanic-Dataset.csv")
   X_train, X_test, y_train, y_test = preprocess_data(df)
   X_train_scaled, X_test_scaled = scale_data(X_train, X_test)

   model = train_random_forest(X_train_scaled, y_train)
   y_pred = predict(model, X_test_scaled)
   metrics = evaluate_model(y_test, y_pred)
   print(metrics)
