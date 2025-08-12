# MLOps Iris Classification Pipeline

A complete MLOps pipeline for Iris classification demonstrating best practices for model development, tracking, deployment, and monitoring.

## Architecture Overview

```
mlops-iris-pipeline/
├── data/                   # Dataset storage
├── src/                    # Source code
│   ├── data_preprocessing.py
│   ├── model_training.py
│   ├── api.py
│   └── monitoring.py
├── models/                 # Trained model artifacts
├── tests/                  # Unit tests
├── docker/                 # Docker configuration
├── scripts/                # Deployment scripts
├── logs/                   # Application logs
├── .github/workflows/      # CI/CD pipelines
└── requirements.txt        # Dependencies
```

## Features

- **Data Versioning**: Clean data preprocessing with version control
- **Experiment Tracking**: MLflow for tracking model experiments and metrics
- **Model Registry**: Centralized model versioning and management
- **REST API**: FastAPI-based prediction service
- **Containerization**: Docker for consistent deployment
- **CI/CD**: GitHub Actions for automated testing and deployment
- **Monitoring**: Logging and metrics collection for model performance
- **Testing**: Comprehensive unit tests

## Quick Start

### 1. Setup Environment
```bash
pip install -r requirements.txt
```

### 2. Train Models
```bash
python src/model_training.py
```

### 3. Start API
```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

### 4. Make Predictions
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}'
```

## API Endpoints

- `GET /health` - Health check
- `POST /predict` - Make predictions
- `GET /metrics` - Prometheus metrics
- `GET /model-info` - Model metadata

## Model Performance

The pipeline trains and compares multiple models:
- Logistic Regression
- Random Forest
- Support Vector Machine

Best performing model is automatically registered in MLflow.

## Monitoring

- Request/response logging
- Model prediction metrics
- Performance monitoring dashboard
- Alerting for model drift

## Deployment

### Local Docker
```bash
docker build -t iris-ml-api -f docker/Dockerfile .
docker run -p 8000:8000 iris-ml-api
```

### CI/CD Pipeline
GitHub Actions automatically:
1. Runs tests and linting
2. Builds Docker image
3. Pushes to Docker Hub
4. Deploys to production

## Technology Stack

- **ML Framework**: Scikit-learn
- **Experiment Tracking**: MLflow
- **API Framework**: FastAPI
- **Containerization**: Docker
- **CI/CD**: GitHub Actions
- **Monitoring**: Prometheus + Custom logging
- **Testing**: pytest
- **Code Quality**: flake8, black

## License

MIT License
