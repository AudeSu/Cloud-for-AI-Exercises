from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

def predict(model, X_test):
    """Make predictions using the trained model."""
    return model.predict(X_test)

def evaluate_model(y_test, y_pred):
    """Evaluate model performance."""
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "confusion_matrix": conf_matrix
    }


# def predict_and_evaluate(models, X_test, y_test):
#     """Evaluates the models and prints metrics."""
#     for model_name, model in models.items():
#         y_pred = model.predict(X_test)
#         print(f"{model_name} Metrics:")
#         print("Accuracy:", accuracy_score(y_test, y_pred))
#         print("Precision:", precision_score(y_test, y_pred))
#         print("Recall:", recall_score(y_test, y_pred))
#         print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
#         print(classification_report(y_test, y_pred))
#         print("-" * 50)