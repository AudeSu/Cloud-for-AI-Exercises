import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(file_path: str) -> pd.DataFrame:
    """Load the Titanic dataset."""
    return pd.read_csv(file_path)

def preprocess_data(df: pd.DataFrame):
    """Clean and preprocess the Titanic dataset."""
    df['Age'].fillna(df['Age'].median(), inplace=True)
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
    df.drop('Cabin', axis=1, inplace=True)

    
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)

    X = df.drop(['Survived', 'Name', 'Ticket', 'PassengerId'], axis=1)
    y = df['Survived']

    return train_test_split(X, y, test_size=0.2, random_state=42)

def scale_data(X_train, X_test):
    """Scale numerical data."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled


# def load_and_preprocess_data(filepath: str):
#     """Loads and preprocesses the Titanic dataset."""
#     df = pd.read_csv(filepath)
    
#     # Handle missing values
#     df['Age'].fillna(df['Age'].median(), inplace=True)
#     df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
#     df.drop('Cabin', axis=1, inplace=True)
    
#     # Encode categorical variables
#     df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
#     df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)
    
#     # Feature and target separation
#     X = df.drop(['Survived', 'Name', 'Ticket', 'PassengerId'], axis=1)
#     y = df['Survived']
    
#     return train_test_split(X, y, test_size=0.2, random_state=42)