# Titanic ML Pipeline (Exercise 2)

This project builds a production-ready ML pipeline for the Titanic survival prediction challenge.

## Structure
- `data/`: Dataset storage.
- `notebooks/`: Jupyter notebooks of the exercises.
- `src/`: Core modules for data processing, model training, and prediction.
- `tests/`: Unit tests for the pipeline.

## Usage
1. Install dependencies: `pip install -r requirements.txt`
2. Run tests: `pytest tests/`
3. Execute the pipeline: `python main.py`


# Titanic Survival Prediction API (Exercise 3)

This project provides a REST API for predicting passenger survival on the Titanic using machine learning. The model is trained on historical data and deployed using FastAPI and Docker.

## Project Structure
```
titanic-survival-prediction/
├── app/
│   └── main.py
├── data/
│   └── Titanic-Dataset.csv
├── models/
│   ├── model.pkl
│   └── scaler.pkl
├── src/
│   ├── data_processing.py
│   ├── model_training.py
│   └── prediction.py
├── Dockerfile
├── requirements.txt
├── save_model.py
└── README.md
```

## Setup and Installation

1. Train and save the model:
```bash
python save_model.py
```

2. Build the Docker image:
```bash
docker build -t titanic-prediction-api .
```

3. Run the container:
```bash
docker run -d -p 8000:8000 titanic-prediction-api
```

## API Usage

The API provides two endpoints:

1. `GET /`: Welcome message
2. `POST /predict`: Make survival predictions

### Making Predictions

Send a POST request to `/predict` with JSON data in the following format:

```json
{
    "Pclass": 3,
    "Sex": 0,
    "Age": 25,
    "SibSp": 0,
    "Parch": 0,
    "Fare": 7.25,
    "Embarked_Q": 0,
    "Embarked_S": 1
}
```

The API will return:
```json
{
    "survival_prediction": 0,
    "survival_probability": 0.234
}
```
