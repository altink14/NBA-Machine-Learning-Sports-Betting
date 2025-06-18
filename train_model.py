import os
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split

def train_and_save_model():
    # Create Models directory if it doesn't exist
    if not os.path.exists('Models'):
        os.makedirs('Models')

    # Create sample training data
    X = np.random.rand(1000, 10)  # 1000 games, 10 features
    y = np.random.randint(0, 2, 1000)  # Binary outcome (0 or 1)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create and train XGBoost classifier
    model = xgb.XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        objective='binary:logistic'
    )
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Save the model
    model.save_model('Models/XGBoost_ML.json')
    print("Model trained and saved successfully!")

if __name__ == "__main__":
    train_and_save_model()