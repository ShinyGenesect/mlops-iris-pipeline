#!/bin/bash

# MLOps Iris Pipeline Deployment Script
# This script handles the complete deployment of the Iris ML API

set -e  # Exit on any error

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
# Go to project root (parent of scripts directory)
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Change to project root directory
cd "$PROJECT_ROOT"

echo "Working from project directory: $(pwd)"

# Configuration
DOCKER_IMAGE_NAME="iris-ml-api"
CONTAINER_NAME="iris-ml-api"
PORT="8000"
HEALTH_CHECK_URL="http://localhost:${PORT}/health"
MAX_HEALTH_CHECKS=30
HEALTH_CHECK_INTERVAL=5

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Function to check if Docker is running
check_docker() {
    log "Checking Docker installation..."
    if ! command -v docker &> /dev/null; then
        error "Docker is not installed. Please install Docker first."
        exit 1
    fi

    if ! docker info &> /dev/null; then
        error "Docker daemon is not running. Please start Docker."
        exit 1
    fi

    success "Docker is running"
}

# Function to stop and remove existing container
cleanup_existing() {
    log "Cleaning up existing container..."

    if docker ps -q -f name=${CONTAINER_NAME} | grep -q .; then
        warning "Stopping existing container: ${CONTAINER_NAME}"
        docker stop ${CONTAINER_NAME}
    fi

    if docker ps -aq -f name=${CONTAINER_NAME} | grep -q .; then
        warning "Removing existing container: ${CONTAINER_NAME}"
        docker rm ${CONTAINER_NAME}
    fi

    success "Cleanup completed"
}

# Function to build Docker image
build_image() {
    log "Building Docker image: ${DOCKER_IMAGE_NAME}"

    if [ ! -f "docker/Dockerfile" ]; then
        error "Dockerfile not found at docker/Dockerfile"
        exit 1
    fi

    docker build -t ${DOCKER_IMAGE_NAME} -f docker/Dockerfile .

    if [ $? -eq 0 ]; then
        success "Docker image built successfully"
    else
        error "Failed to build Docker image"
        exit 1
    fi
}

# Function to run the container
run_container() {
    log "Starting container: ${CONTAINER_NAME}"

    # Create necessary directories
    mkdir -p logs models data

    docker run -d \
        --name ${CONTAINER_NAME} \
        -p ${PORT}:8000 \
        -v "$(pwd)/logs:/app/logs" \
        -v "$(pwd)/models:/app/models" \
        -v "$(pwd)/data:/app/data" \
        --restart unless-stopped \
        ${DOCKER_IMAGE_NAME}

    if [ $? -eq 0 ]; then
        success "Container started successfully"
        log "Container ID: $(docker ps -q -f name=${CONTAINER_NAME})"
    else
        error "Failed to start container"
        exit 1
    fi
}

# Function to wait for health check
wait_for_health() {
    log "Waiting for application to be healthy..."

    for i in $(seq 1 $MAX_HEALTH_CHECKS); do
        if curl -f ${HEALTH_CHECK_URL} &> /dev/null; then
            success "Application is healthy!"
            return 0
        fi

        log "Health check attempt ${i}/${MAX_HEALTH_CHECKS} failed. Waiting ${HEALTH_CHECK_INTERVAL}s..."
        sleep $HEALTH_CHECK_INTERVAL
    done

    error "Application failed to become healthy after ${MAX_HEALTH_CHECKS} attempts"

    # Show container logs for debugging
    log "Container logs:"
    docker logs ${CONTAINER_NAME}

    return 1
}

# Function to show deployment info
show_deployment_info() {
    log "Deployment Information:"
    echo "=================================="
    echo "Container Name: ${CONTAINER_NAME}"
    echo "Docker Image: ${DOCKER_IMAGE_NAME}"
    echo "Port: ${PORT}"
    echo "API URL: http://localhost:${PORT}"
    echo "Health Check: ${HEALTH_CHECK_URL}"
    echo "API Documentation: http://localhost:${PORT}/docs"
    echo "Metrics: http://localhost:${PORT}/metrics"
    echo "=================================="

    log "Testing API endpoints:"
    echo ""

    # Test root endpoint
    echo "1. Root endpoint:"
    curl -s http://localhost:${PORT}/ | python3 -m json.tool || echo "Failed to reach root endpoint"
    echo ""

    # Test model info
    echo "2. Model info endpoint:"
    curl -s http://localhost:${PORT}/model-info | python3 -m json.tool || echo "Failed to reach model-info endpoint"
    echo ""

    log "Container status:"
    docker ps -f name=${CONTAINER_NAME}
}

# Function to run a prediction test
test_prediction() {
    log "Testing prediction endpoint..."

    cat << 'EOF' > test_prediction.json
{
    "sepal_length": 5.1,
    "sepal_width": 3.5,
    "petal_length": 1.4,
    "petal_width": 0.2
}
EOF

    echo "Test input:"
    cat test_prediction.json
    echo ""

    echo "Prediction result:"
    curl -s -X POST \
        -H "Content-Type: application/json" \
        -d @test_prediction.json \
        http://localhost:${PORT}/predict | python3 -m json.tool || echo "Prediction test failed"

    rm -f test_prediction.json
}

# Main deployment function
deploy() {
    log "Starting MLOps Iris Pipeline Deployment"

    check_docker
    cleanup_existing
    build_image
    run_container

    if wait_for_health; then
        show_deployment_info
        test_prediction
        success "Deployment completed successfully!"

        log "To view logs: docker logs -f ${CONTAINER_NAME}"
        log "To stop: docker stop ${CONTAINER_NAME}"
        log "To restart: docker restart ${CONTAINER_NAME}"
    else
        error "Deployment failed - application is not healthy"
        exit 1
    fi
}

# Function to stop the deployment
stop_deployment() {
    log "Stopping deployment..."
    cleanup_existing
    success "Deployment stopped"
}

# Function to show logs
show_logs() {
    if docker ps -q -f name=${CONTAINER_NAME} | grep -q .; then
        docker logs -f ${CONTAINER_NAME}
    else
        error "Container ${CONTAINER_NAME} is not running"
        exit 1
    fi
}

# Function to show status
show_status() {
    log "Deployment Status:"

    if docker ps -q -f name=${CONTAINER_NAME} | grep -q .; then
        success "Container is running"
        docker ps -f name=${CONTAINER_NAME}

        log "Health check:"
        if curl -f ${HEALTH_CHECK_URL} &> /dev/null; then
            success "Application is healthy"
        else
            warning "Application health check failed"
        fi
    else
        warning "Container is not running"
    fi
}

# Main script logic
case "${1:-deploy}" in
    "deploy")
        deploy
        ;;
    "stop")
        stop_deployment
        ;;
    "logs")
        show_logs
        ;;
    "status")
        show_status
        ;;
    "restart")
        stop_deployment
        deploy
        ;;
    *)
        echo "Usage: $0 {deploy|stop|logs|status|restart}"
        echo ""
        echo "Commands:"
        echo "  deploy   - Build and deploy the application (default)"
        echo "  stop     - Stop and remove the container"
        echo "  logs     - Show container logs (follow mode)"
        echo "  status   - Show deployment status"
        echo "  restart  - Stop and redeploy the application"
        exit 1
        ;;
esac
