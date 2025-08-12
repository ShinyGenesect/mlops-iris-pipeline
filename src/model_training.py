import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import mlflow
import mlflow.sklearn
import joblib
import os
import logging
from data_preprocessing import DataPreprocessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(self, experiment_name="iris_classification"):
        self.experiment_name = experiment_name
        self.models = {
            "logistic_regression": LogisticRegression(random_state=42, max_iter=1000),
            "random_forest": RandomForestClassifier(random_state=42, n_estimators=100),
            "svm": SVC(random_state=42, probability=True)
        }
        self.best_model = None
        self.best_metrics = None
        self.target_names = ['setosa', 'versicolor', 'virginica']

        # Setup MLflow
        mlflow.set_experiment(experiment_name)

    def evaluate_model(self, model, X_test, y_test):
        """Evaluate model and return metrics"""
        y_pred = model.predict(X_test)

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, average='weighted'),
            "recall": recall_score(y_test, y_pred, average='weighted'),
            "f1_score": f1_score(y_test, y_pred, average='weighted')
        }

        return metrics, y_pred

    def train_and_evaluate(self, X_train, X_test, y_train, y_test):
        """Train all models and track experiments with MLflow"""
        logger.info("Starting model training and evaluation...")

        all_results = {}

        for model_name, model in self.models.items():
            logger.info(f"Training {model_name}...")

            with mlflow.start_run(run_name=model_name):
                # Train model
                model.fit(X_train, y_train)

                # Evaluate model
                metrics, y_pred = self.evaluate_model(model, X_test, y_test)

                # Log parameters
                if hasattr(model, 'get_params'):
                    params = model.get_params()
                    for param, value in params.items():
                        mlflow.log_param(param, value)

                # Log metrics
                for metric_name, value in metrics.items():
                    mlflow.log_metric(metric_name, value)

                # Log model
                mlflow.sklearn.log_model(
                    model,
                    model_name,
                    registered_model_name=f"iris_{model_name}"
                )

                # Generate and log classification report
                report = classification_report(y_test, y_pred, target_names=self.target_names)
                mlflow.log_text(report, f"{model_name}_classification_report.txt")

                all_results[model_name] = {
                    "model": model,
                    "metrics": metrics,
                    "run_id": mlflow.active_run().info.run_id
                }

                logger.info(f"{model_name} - Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1_score']:.4f}")

        return all_results

    def select_best_model(self, results):
        """Select the best model based on F1 score"""
        best_model_name = max(results.keys(), key=lambda k: results[k]["metrics"]["f1_score"])

        self.best_model = results[best_model_name]["model"]
        self.best_metrics = results[best_model_name]["metrics"]

        logger.info(f"Best model: {best_model_name}")
        logger.info(f"Best metrics: {self.best_metrics}")

        return best_model_name, self.best_model

    def save_model(self, model, model_name, filepath):
        """Save the trained model"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(model, filepath)

        # Also save model metadata
        metadata = {
            "model_name": model_name,
            "metrics": self.best_metrics,
            "target_names": self.target_names,
            "feature_names": ["sepal_length", "sepal_width", "petal_length", "petal_width"]
        }

        metadata_path = filepath.replace('.pkl', '_metadata.pkl')
        joblib.dump(metadata, metadata_path)

        logger.info(f"Model saved to {filepath}")
        logger.info(f"Metadata saved to {metadata_path}")

    def register_best_model(self, model_name):
        """Register the best model in MLflow Model Registry"""
        try:
            # Get the latest version of the model
            client = mlflow.tracking.MlflowClient()
            model_name_full = f"iris_{model_name}"

            # Transition to Production stage
            latest_version = client.get_latest_versions(model_name_full, stages=["None"])[0]

            client.transition_model_version_stage(
                name=model_name_full,
                version=latest_version.version,
                stage="Production"
            )

            logger.info(f"Model {model_name_full} version {latest_version.version} registered as Production")

        except Exception as e:
            logger.warning(f"Could not register model: {e}")

def main():
    """Main training pipeline"""
    logger.info("Starting MLOps training pipeline...")

    # Initialize components
    preprocessor = DataPreprocessor()
    trainer = ModelTrainer()

    # Load and preprocess data
    df = preprocessor.load_data()
    X_train, X_test, y_train, y_test = preprocessor.preprocess_data(df)

    # Save preprocessor
    preprocessor.save_preprocessor("models/scaler.pkl")

    # Train and evaluate models
    results = trainer.train_and_evaluate(X_train, X_test, y_train, y_test)

    # Select best model
    best_model_name, best_model = trainer.select_best_model(results)

    # Save best model
    trainer.save_model(best_model, best_model_name, "models/best_model.pkl")

    # Register in MLflow
    trainer.register_best_model(best_model_name)

    # Save processed data for future use
    np.savez("data/processed_data.npz",
             X_train=X_train, X_test=X_test,
             y_train=y_train, y_test=y_test)

    # Save raw data
    df.to_csv("data/iris_dataset.csv", index=False)

    logger.info("Training pipeline completed successfully!")
    logger.info(f"MLflow tracking UI: mlflow ui")
    logger.info(f"Best model saved as: models/best_model.pkl")

if __name__ == "__main__":
    main()
