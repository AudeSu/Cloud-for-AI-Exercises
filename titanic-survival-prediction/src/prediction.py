from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "confusion_matrix": confusion_matrix(y_test, y_pred)
    }
    return metrics


def print_metrics(metrics):
    print("\nModel Evaluation Metrics:")
    for key, value in metrics.items():
        print(f"{key.capitalize()}: {value}")
