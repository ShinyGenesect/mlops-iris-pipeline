import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_names = None
        self.target_names = None

    def load_data(self):
        """Load the Iris dataset from sklearn"""
        logger.info("Loading Iris dataset...")
        iris = load_iris()

        # Create DataFrame for better handling
        df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
        df['target'] = iris.target

        self.feature_names = iris.feature_names
        self.target_names = iris.target_names

        logger.info(f"Dataset loaded: {df.shape[0]} samples, {df.shape[1]-1} features")
        logger.info(f"Features: {list(self.feature_names)}")
        logger.info(f"Target classes: {list(self.target_names)}")

        return df

    def preprocess_data(self, df, test_size=0.2, random_state=42):
        """Preprocess the data: split and scale"""
        logger.info("Preprocessing data...")

        # Separate features and target
        X = df.drop('target', axis=1)
        y = df['target']

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        # Scale the features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        logger.info(f"Train set: {X_train_scaled.shape[0]} samples")
        logger.info(f"Test set: {X_test_scaled.shape[0]} samples")

        return X_train_scaled, X_test_scaled, y_train, y_test

    def save_preprocessor(self, filepath):
        """Save the preprocessor (scaler) for later use"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self.scaler, filepath)
        logger.info(f"Preprocessor saved to {filepath}")

    def load_preprocessor(self, filepath):
        """Load a saved preprocessor"""
        self.scaler = joblib.load(filepath)
        logger.info(f"Preprocessor loaded from {filepath}")

    def transform_new_data(self, data):
        """Transform new data using the fitted scaler"""
        if isinstance(data, dict):
            # Convert dict to DataFrame for consistency
            data = pd.DataFrame([data])

        return self.scaler.transform(data)

if __name__ == "__main__":
    # Example usage
    preprocessor = DataPreprocessor()

    # Load and preprocess data
    df = preprocessor.load_data()
    X_train, X_test, y_train, y_test = preprocessor.preprocess_data(df)

    # Save preprocessor
    preprocessor.save_preprocessor("models/scaler.pkl")

    # Save processed data
    os.makedirs("data", exist_ok=True)
    np.savez("data/processed_data.npz",
             X_train=X_train, X_test=X_test,
             y_train=y_train, y_test=y_test)

    # Save raw data as CSV for reference
    df.to_csv("data/iris_dataset.csv", index=False)

    logger.info("Data preprocessing completed!")
    logger.info(f"Raw data saved to: data/iris_dataset.csv")
    logger.info(f"Processed data saved to: data/processed_data.npz")
    logger.info(f"Scaler saved to: models/scaler.pkl")
