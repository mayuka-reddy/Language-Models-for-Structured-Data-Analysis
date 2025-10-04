# Makefile for NL-to-SQL Assistant

.PHONY: help install test format lint run-ui clean setup dev-install

# Default target
help:
	@echo "Available commands:"
	@echo "  setup       - Complete setup (install + create sample data)"
	@echo "  install     - Install dependencies"
	@echo "  dev-install - Install development dependencies"
	@echo "  test        - Run tests"
	@echo "  format      - Format code with black"
	@echo "  lint        - Lint code with flake8"
	@echo "  run-ui      - Start Gradio UI"
	@echo "  clean       - Clean temporary files"

# Setup everything
setup: install create-data
	@echo "✅ Setup complete! Run 'make run-ui' to start the application."

# Install dependencies
install:
	python install_dependencies.py

# Alternative: install from requirements file
install-from-file:
	pip install -r requirements_minimal.txt

# Install development dependencies
dev-install: install
	pip install pytest black flake8 jupyter

# Create sample data
create-data:
	@echo "Creating sample database..."
	@python -c "from app.sql_executor import SQLExecutor; SQLExecutor()"
	@echo "✅ Sample database created"

# Run tests
test:
	pytest tests/ -v

# Format code
format:
	black app/ ui/ tests/ --line-length 88

# Lint code
lint:
	flake8 app/ ui/ tests/ --max-line-length=88 --ignore=E203,W503

# Run Gradio UI
run-ui:
	@echo "Starting NL-to-SQL Assistant UI..."
	@echo "Open http://localhost:7860 in your browser"
	python ui/gradio_app.py

# Clean temporary files
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf .pytest_cache/
	rm -rf build/
	rm -rf dist/

# Development workflow
dev: dev-install format lint test
	@echo "✅ Development checks passed"

# Quick test run
quick-test:
	pytest tests/test_inference.py -v

# Install and run (for first-time users)
start: setup run-ui

# Check system requirements
check-requirements:
	@echo "Checking Python version..."
	@python --version
	@echo "Checking pip..."
	@pip --version
	@echo "Checking available space..."
	@df -h .
	@echo "✅ System check complete"

# Download models (optional)
download-models:
	@echo "Pre-downloading models..."
	@python -c "from transformers import T5ForConditionalGeneration, T5Tokenizer; T5ForConditionalGeneration.from_pretrained('t5-small'); T5Tokenizer.from_pretrained('t5-small')"
	@echo "✅ Models downloaded"

# Run specific tests
test-inference:
	pytest tests/test_inference.py -v

test-executor:
	pytest tests/test_sql_executor.py -v

test-metrics:
	pytest tests/test_metrics.py -v

# Generate test coverage report
coverage:
	pytest tests/ --cov=app --cov-report=html
	@echo "Coverage report generated in htmlcov/"

# Build documentation (if needed)
docs:
	@echo "Documentation available in README.md"

# Docker commands (if Docker is available)
docker-build:
	docker build -t nl2sql-assistant .

docker-run:
	docker run -p 7860:7860 nl2sql-assistant

# Jupyter notebook server
notebook:
	jupyter notebook notebooks/

# Performance benchmark
benchmark:
	@echo "Running performance benchmark..."
	@python -c "
import time
from app.inference import NL2SQLInference
from app.sql_executor import SQLExecutor

print('Initializing components...')
inference = NL2SQLInference()
executor = SQLExecutor()

print('Running benchmark...')
start_time = time.time()

for i in range(10):
    result = inference.generate_sql('Show me all customers')
    exec_result = executor.execute_query(result['sql'])

end_time = time.time()
avg_time = (end_time - start_time) / 10

print(f'Average time per query: {avg_time:.3f}s')
print('✅ Benchmark complete')
"