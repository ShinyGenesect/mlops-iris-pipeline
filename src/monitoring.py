import logging
import json
import sqlite3
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import pandas as pd
from dataclasses import dataclass
from prometheus_client import Counter, Histogram, Gauge, Info
import threading
import time

# Prometheus metrics for monitoring
PREDICTION_COUNTER = Counter('ml_predictions_total', 'Total predictions made', ['model_name', 'predicted_class'])
PREDICTION_LATENCY = Histogram('ml_prediction_duration_seconds', 'Prediction latency', ['model_name'])
MODEL_ACCURACY = Gauge('ml_model_accuracy', 'Model accuracy', ['model_name'])
ACTIVE_REQUESTS = Gauge('ml_active_requests', 'Number of active prediction requests')
MODEL_INFO = Info('ml_model_info', 'Information about the loaded model')

@dataclass
class PredictionLog:
    """Data class for prediction logs"""
    timestamp: str
    input_data: Dict
    prediction: str
    confidence: float
    model_version: str
    response_time: float
    user_id: Optional[str] = None
    session_id: Optional[str] = None

class DatabaseLogger:
    """SQLite database logger for storing prediction logs"""

    def __init__(self, db_path: str = "logs/predictions.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.init_database()

    def init_database(self):
        """Initialize the database with required tables"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    input_data TEXT NOT NULL,
                    prediction TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    model_version TEXT NOT NULL,
                    response_time REAL NOT NULL,
                    user_id TEXT,
                    session_id TEXT
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS model_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    metric_name TEXT NOT NULL,
                    metric_value REAL NOT NULL
                )
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_predictions_timestamp
                ON predictions(timestamp)
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_metrics_timestamp
                ON model_metrics(timestamp)
            """)

    def log_prediction(self, prediction_log: PredictionLog):
        """Log a prediction to the database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO predictions
                (timestamp, input_data, prediction, confidence, model_version,
                 response_time, user_id, session_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                prediction_log.timestamp,
                json.dumps(prediction_log.input_data),
                prediction_log.prediction,
                prediction_log.confidence,
                prediction_log.model_version,
                prediction_log.response_time,
                prediction_log.user_id,
                prediction_log.session_id
            ))

    def log_metric(self, model_name: str, metric_name: str, metric_value: float):
        """Log a model metric to the database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO model_metrics (timestamp, model_name, metric_name, metric_value)
                VALUES (?, ?, ?, ?)
            """, (
                datetime.now().isoformat(),
                model_name,
                metric_name,
                metric_value
            ))

    def get_predictions(self, limit: int = 100, since: Optional[datetime] = None) -> List[Dict]:
        """Retrieve recent predictions from the database"""
        with sqlite3.connect(self.db_path) as conn:
            if since:
                cursor = conn.execute("""
                    SELECT * FROM predictions
                    WHERE timestamp > ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                """, (since.isoformat(), limit))
            else:
                cursor = conn.execute("""
                    SELECT * FROM predictions
                    ORDER BY timestamp DESC
                    LIMIT ?
                """, (limit,))

            columns = [description[0] for description in cursor.description]
            return [dict(zip(columns, row)) for row in cursor.fetchall()]

    def get_prediction_stats(self, hours: int = 24) -> Dict:
        """Get prediction statistics for the last N hours"""
        since = datetime.now() - timedelta(hours=hours)

        with sqlite3.connect(self.db_path) as conn:
            # Total predictions
            total_predictions = conn.execute("""
                SELECT COUNT(*) FROM predictions
                WHERE timestamp > ?
            """, (since.isoformat(),)).fetchone()[0]

            # Predictions by class
            class_counts = conn.execute("""
                SELECT prediction, COUNT(*)
                FROM predictions
                WHERE timestamp > ?
                GROUP BY prediction
            """, (since.isoformat(),)).fetchall()

            # Average confidence
            avg_confidence = conn.execute("""
                SELECT AVG(confidence)
                FROM predictions
                WHERE timestamp > ?
            """, (since.isoformat(),)).fetchone()[0]

            # Average response time
            avg_response_time = conn.execute("""
                SELECT AVG(response_time)
                FROM predictions
                WHERE timestamp > ?
            """, (since.isoformat(),)).fetchone()[0]

            return {
                "total_predictions": total_predictions,
                "class_distribution": dict(class_counts),
                "average_confidence": avg_confidence or 0,
                "average_response_time": avg_response_time or 0,
                "time_period_hours": hours
            }

class ModelMonitor:
    """Main monitoring class for the ML model"""

    def __init__(self, model_name: str = "iris_classifier"):
        self.model_name = model_name
        self.db_logger = DatabaseLogger()
        self.setup_logging()

        # Performance tracking
        self.prediction_count = 0
        self.total_response_time = 0.0
        self.confidence_scores = []

        # Start background monitoring
        self.monitoring_thread = threading.Thread(target=self._background_monitoring, daemon=True)
        self.monitoring_thread.start()

    def setup_logging(self):
        """Setup file logging for the monitor"""
        os.makedirs("logs", exist_ok=True)

        # Create monitor-specific logger
        self.logger = logging.getLogger(f"model_monitor_{self.model_name}")
        self.logger.setLevel(logging.INFO)

        # File handler
        file_handler = logging.FileHandler(f"logs/model_monitor.log")
        file_handler.setLevel(logging.INFO)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # Add handlers if not already added
        if not self.logger.handlers:
            self.logger.addHandler(file_handler)
            self.logger.addHandler(console_handler)

    def log_prediction(self, input_data: Dict, prediction: str, confidence: float,
                      response_time: float, model_version: str = "1.0.0",
                      user_id: Optional[str] = None, session_id: Optional[str] = None):
        """Log a prediction with all relevant information"""

        # Create prediction log
        prediction_log = PredictionLog(
            timestamp=datetime.now().isoformat(),
            input_data=input_data,
            prediction=prediction,
            confidence=confidence,
            model_version=model_version,
            response_time=response_time,
            user_id=user_id,
            session_id=session_id
        )

        # Store in database
        self.db_logger.log_prediction(prediction_log)

        # Update metrics
        self.prediction_count += 1
        self.total_response_time += response_time
        self.confidence_scores.append(confidence)

        # Update Prometheus metrics
        PREDICTION_COUNTER.labels(
            model_name=self.model_name,
            predicted_class=prediction
        ).inc()

        PREDICTION_LATENCY.labels(model_name=self.model_name).observe(response_time)

        # Log to file
        self.logger.info(f"Prediction logged: {prediction} (confidence: {confidence:.3f}, "
                        f"response_time: {response_time:.3f}s)")

    def get_model_performance(self) -> Dict:
        """Get current model performance metrics"""
        if self.prediction_count == 0:
            return {
                "predictions_count": 0,
                "average_response_time": 0,
                "average_confidence": 0
            }

        return {
            "predictions_count": self.prediction_count,
            "average_response_time": self.total_response_time / self.prediction_count,
            "average_confidence": sum(self.confidence_scores) / len(self.confidence_scores),
            "min_confidence": min(self.confidence_scores),
            "max_confidence": max(self.confidence_scores)
        }

    def get_recent_predictions(self, limit: int = 50) -> List[Dict]:
        """Get recent predictions from the database"""
        return self.db_logger.get_predictions(limit=limit)

    def get_prediction_statistics(self, hours: int = 24) -> Dict:
        """Get prediction statistics for the specified time period"""
        return self.db_logger.get_prediction_stats(hours=hours)

    def detect_data_drift(self, recent_hours: int = 24) -> Dict:
        """Simple data drift detection based on prediction distribution"""
        stats = self.get_prediction_statistics(hours=recent_hours)

        # Expected distribution for Iris dataset (approximately equal)
        expected_distribution = {"setosa": 0.33, "versicolor": 0.33, "virginica": 0.34}

        actual_distribution = stats["class_distribution"]
        total_predictions = stats["total_predictions"]

        if total_predictions == 0:
            return {"drift_detected": False, "reason": "No predictions available"}

        # Normalize actual distribution
        actual_normalized = {
            class_name: count / total_predictions
            for class_name, count in actual_distribution.items()
        }

        # Calculate drift score (simple chi-square like measure)
        drift_score = 0
        for class_name in expected_distribution:
            expected = expected_distribution[class_name]
            actual = actual_normalized.get(class_name, 0)
            drift_score += abs(expected - actual)

        # Threshold for drift detection
        drift_threshold = 0.3  # 30% deviation
        drift_detected = drift_score > drift_threshold

        return {
            "drift_detected": drift_detected,
            "drift_score": drift_score,
            "threshold": drift_threshold,
            "expected_distribution": expected_distribution,
            "actual_distribution": actual_normalized,
            "total_predictions": total_predictions
        }

    def _background_monitoring(self):
        """Background thread for periodic monitoring tasks"""
        while True:
            try:
                # Update model accuracy metric
                performance = self.get_model_performance()
                if performance["predictions_count"] > 0:
                    MODEL_ACCURACY.labels(model_name=self.model_name).set(
                        performance["average_confidence"]
                    )

                # Check for data drift every hour
                drift_info = self.detect_data_drift()
                if drift_info["drift_detected"]:
                    self.logger.warning(f"Data drift detected: {drift_info}")

                # Sleep for 5 minutes
                time.sleep(300)

            except Exception as e:
                self.logger.error(f"Background monitoring error: {e}")
                time.sleep(60)  # Wait 1 minute before retrying

    def export_logs_to_csv(self, filepath: str, hours: int = 24):
        """Export recent logs to CSV format"""
        since = datetime.now() - timedelta(hours=hours)
        predictions = self.db_logger.get_predictions(limit=10000, since=since)

        if predictions:
            df = pd.DataFrame(predictions)
            df.to_csv(filepath, index=False)
            self.logger.info(f"Exported {len(predictions)} predictions to {filepath}")
        else:
            self.logger.info("No predictions to export")

# Global monitor instance
_monitor_instance = None

def get_monitor(model_name: str = "iris_classifier") -> ModelMonitor:
    """Get or create the global monitor instance"""
    global _monitor_instance
    if _monitor_instance is None:
        _monitor_instance = ModelMonitor(model_name)
    return _monitor_instance

# Context manager for tracking requests
class RequestTracker:
    """Context manager for tracking API requests"""

    def __init__(self, monitor: ModelMonitor):
        self.monitor = monitor
        self.start_time = None

    def __enter__(self):
        ACTIVE_REQUESTS.inc()
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        ACTIVE_REQUESTS.dec()
        if self.start_time:
            duration = time.time() - self.start_time
            return duration
        return 0
