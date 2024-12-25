from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

def train_random_forest(X_train, y_train):
    """Train a Random Forest classifier."""
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_train, y_train)
    return rf

def tune_random_forest(X_train, y_train):
    """Perform hyperparameter tuning on Random Forest."""
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    grid_rf = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, scoring='accuracy')
    grid_rf.fit(X_train, y_train)
    return grid_rf.best_estimator_


# def train_model(X_train, y_train):
#     """Trains Logistic Regression and Random Forest models."""
#     # Logistic Regression
#     logreg = LogisticRegression(random_state=42, max_iter=1000)
#     logreg.fit(X_train, y_train)
    
#     # Random Forest with hyperparameter tuning
#     param_grid = {
#         'n_estimators': [50, 100, 200],
#         'max_depth': [None, 10, 20],
#         'min_samples_split': [2, 5, 10],
#         'min_samples_leaf': [1, 2, 4]
#     }
#     grid_rf = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, scoring='accuracy')
#     grid_rf.fit(X_train, y_train)
#     best_rf = grid_rf.best_estimator_
    
#     return logreg, best_rf