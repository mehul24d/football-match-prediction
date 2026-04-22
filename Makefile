.PHONY: install install-dev lint format test test-cov run-api docker-build docker-up clean

## Install production dependencies
install:
	pip install -r requirements.txt

## Install package in editable mode (development)
install-dev:
	pip install -e ".[dev]"
	pip install -r requirements.txt

## Lint with flake8
lint:
	flake8 src/ api/ tests/ --max-line-length=100 --ignore=E203,W503

## Format with black + isort
format:
	isort src/ api/ tests/
	black src/ api/ tests/ --line-length=100

## Run tests
test:
	pytest tests/ -v

## Run tests with coverage report
test-cov:
	pytest tests/ -v --cov=src --cov=api --cov-report=term-missing --cov-report=html

## Run the FastAPI development server
run-api:
	uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

## Build Docker image
docker-build:
	docker build -t football-match-prediction:latest -f docker/Dockerfile .

## Start all services with Docker Compose
docker-up:
	docker-compose -f docker/docker-compose.yml up --build

## Stop Docker services
docker-down:
	docker-compose -f docker/docker-compose.yml down

## Run full data pipeline (ingest → preprocess → feature engineering)
pipeline:
	python -m src.data.pipeline

## Train all models and log to MLflow
train:
	python -m src.models.train

## Clean up cache and build artifacts
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/ dist/ htmlcov/ .coverage
