from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import numpy as np
import pandas as pd
import logging
import os
from datetime import datetime
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response
import json

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/api.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Prometheus metrics
REQUEST_COUNT = Counter('api_requests_total', 'Total API requests', ['method', 'endpoint'])
REQUEST_DURATION = Histogram('api_request_duration_seconds', 'API request duration')
PREDICTION_COUNT = Counter('predictions_total', 'Total predictions made', ['predicted_class'])

app = FastAPI(title="Iris Classification API", version="1.0.0")

class IrisInput(BaseModel):
    sepal_length: float = Field(..., description="Sepal length in cm", ge=0, le=10)
    sepal_width: float = Field(..., description="Sepal width in cm", ge=0, le=10)
    petal_length: float = Field(..., description="Petal length in cm", ge=0, le=10)
    petal_width: float = Field(..., description="Petal width in cm", ge=0, le=10)

class PredictionResponse(BaseModel):
    predicted_class: str
    predicted_class_id: int
    confidence_scores: dict
    model_version: str
    timestamp: str

class ModelManager:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.metadata = None
        self.load_model()

    def load_model(self):
        """Load the trained model and preprocessor"""
        try:
            model_path = "models/best_model.pkl"
            scaler_path = "models/scaler.pkl"
            metadata_path = "models/best_model_metadata.pkl"

            if not os.path.exists(model_path):
                logger.error(f"Model file not found: {model_path}")
                raise FileNotFoundError(f"Model file not found: {model_path}")

            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)

            if os.path.exists(metadata_path):
                self.metadata = joblib.load(metadata_path)
            else:
                # Default metadata if file doesn't exist
                self.metadata = {
                    "model_name": "iris_classifier",
                    "target_names": ['setosa', 'versicolor', 'virginica'],
                    "feature_names": ["sepal_length", "sepal_width", "petal_length", "petal_width"]
                }

            logger.info("Model and preprocessor loaded successfully")

        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

    def predict(self, input_data: IrisInput):
        """Make prediction on input data"""
        try:
            # Convert input to DataFrame with the correct feature names that match training
            features = pd.DataFrame([{
                "sepal length (cm)": input_data.sepal_length,
                "sepal width (cm)": input_data.sepal_width,
                "petal length (cm)": input_data.petal_length,
                "petal width (cm)": input_data.petal_width
            }])

            # Scale features
            features_scaled = self.scaler.transform(features)

            # Make prediction
            prediction = self.model.predict(features_scaled)[0]
            prediction_proba = self.model.predict_proba(features_scaled)[0]

            # Prepare response
            predicted_class = self.metadata["target_names"][prediction]
            confidence_scores = {
                self.metadata["target_names"][i]: float(prob)
                for i, prob in enumerate(prediction_proba)
            }

            return PredictionResponse(
                predicted_class=predicted_class,
                predicted_class_id=int(prediction),
                confidence_scores=confidence_scores,
                model_version=self.metadata.get("model_name", "unknown"),
                timestamp=datetime.now().isoformat()
            )

        except Exception as e:
            logger.error(f"Prediction error: {e}")
            raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

# Initialize model manager
model_manager = ModelManager()

@app.middleware("http")
async def log_requests(request, call_next):
    """Middleware to log requests and collect metrics"""
    start_time = datetime.now()

    # Process request
    response = await call_next(request)

    # Calculate duration
    duration = (datetime.now() - start_time).total_seconds()

    # Log request
    logger.info(f"{request.method} {request.url} - {response.status_code} - {duration:.3f}s")

    # Update metrics
    REQUEST_COUNT.labels(method=request.method, endpoint=str(request.url.path)).inc()
    REQUEST_DURATION.observe(duration)

    return response

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Iris Classification API", "version": "1.0.0", "status": "healthy"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Basic health check - ensure model is loaded
        if model_manager.model is None:
            raise HTTPException(status_code=503, detail="Model not loaded")

        return {
            "status": "healthy",
            "model_loaded": True,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Service unhealthy")

@app.post("/predict", response_model=PredictionResponse)
async def predict(input_data: IrisInput):
    """Make prediction on iris features"""
    try:
        # Log the prediction request
        logger.info(f"Prediction request: {input_data.dict()}")

        # Make prediction
        result = model_manager.predict(input_data)

        # Update prediction metrics
        PREDICTION_COUNT.labels(predicted_class=result.predicted_class).inc()

        # Log the prediction result
        logger.info(f"Prediction result: {result.predicted_class} (confidence: {max(result.confidence_scores.values()):.3f})")

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction endpoint error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/model-info")
async def get_model_info():
    """Get model information and metadata"""
    try:
        info = {
            "model_metadata": model_manager.metadata,
            "model_type": str(type(model_manager.model).__name__),
            "feature_names": model_manager.metadata.get("feature_names", []),
            "target_names": model_manager.metadata.get("target_names", []),
            "model_loaded": model_manager.model is not None,
            "scaler_loaded": model_manager.scaler is not None
        }
        return info
    except Exception as e:
        logger.error(f"Model info error: {e}")
        raise HTTPException(status_code=500, detail="Could not retrieve model info")

@app.get("/metrics")
async def get_metrics():
    """Prometheus metrics endpoint"""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.get("/predictions/batch")
async def batch_predict(predictions_data: list[IrisInput]):
    """Batch prediction endpoint"""
    try:
        results = []
        for input_data in predictions_data:
            result = model_manager.predict(input_data)
            results.append(result)
            PREDICTION_COUNT.labels(predicted_class=result.predicted_class).inc()

        logger.info(f"Batch prediction completed for {len(predictions_data)} samples")
        return {"predictions": results, "count": len(results)}

    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail="Batch prediction failed")

if __name__ == "__main__":
    import uvicorn

    # Ensure logs directory exists
    os.makedirs("logs", exist_ok=True)

    # Start the server
    uvicorn.run(app, host="0.0.0.0", port=8000)
