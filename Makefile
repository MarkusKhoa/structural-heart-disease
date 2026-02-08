# Makefile for Structural Heart Disease Joint Embedding Project
# Simplifies common development tasks

.PHONY: help install install-dev test format lint clean train

# Default target
help:
	@echo "Available commands:"
	@echo ""
	@echo "Setup:"
	@echo "  make install        - Install package in editable mode"
	@echo "  make install-dev    - Install with development dependencies"
	@echo ""
	@echo "Development:"
	@echo "  make test           - Run tests with coverage"
	@echo "  make format         - Format code with black and isort"
	@echo "  make lint           - Run linters (flake8, mypy)"
	@echo "  make check          - Run all checks (format, lint, test)"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean          - Remove build artifacts and cache"
	@echo "  make clean-all      - Remove everything including venv"
	@echo "  make clean-checkpoints - Remove training artifacts"
	@echo "  make clean-interactive - Interactive cleanup (recommended)"
	@echo ""
	@echo "Training:"
	@echo "  make train          - Run training with default parameters"
	@echo "  make example        - Run example usage script"
	@echo ""
	@echo "Maintenance:"
	@echo "  make update         - Update all dependencies"
	@echo "  make freeze         - Freeze dependencies to lock file"

# Installation
install:
	pip install -e .

install-dev:
	pip install -r requirements-dev.txt
	pip install -e .
	@echo "✓ Development environment ready!"

# Testing
test:
	pytest --cov=. --cov-report=html --cov-report=term-missing

test-fast:
	pytest -x

# Code quality
format:
	@echo "Formatting code with black..."
	black .
	@echo "Sorting imports with isort..."
	isort .
	@echo "✓ Code formatted!"

lint:
	@echo "Running flake8..."
	flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
	@echo "Running mypy..."
	mypy . --ignore-missing-imports
	@echo "✓ Linting complete!"

check: format lint test
	@echo "✓ All checks passed!"

# Cleaning
clean:
	@echo "Cleaning build artifacts..."
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	@echo "✓ Cleaned!"

clean-all: clean clean-checkpoints
	@echo "Cleaning everything..."
	rm -rf venv/
	@echo "✓ Removed all build artifacts and environments!"

clean-checkpoints:
	rm -rf checkpoints/
	rm -rf logs/
	rm -rf wandb/
	rm -rf runs/
	@echo "✓ Removed training artifacts!"

clean-interactive:
	@echo "Running interactive cleanup..."
	./cleanup.sh

# Training and examples
train:
	python train.py --data_dir ./echonext_dataset --epochs 50 --batch_size 32

example:
	python example_usage.py

# Dependency management
update:
	@echo "Updating dependencies..."
	pip install --upgrade pip
	pip install -r requirements.txt --upgrade
	pip install -r requirements-dev.txt --upgrade
	@echo "✓ Dependencies updated!"

freeze:
	pip freeze > requirements-lock.txt
	@echo "✓ Dependencies frozen to requirements-lock.txt"

# Documentation
docs:
	@echo "Opening documentation..."
	@echo "Main README: README.md"
	@echo "Development Guide: DEVELOPMENT.md"
	@echo "Technical Docs: JOINT_EMBEDDING_README.md"

# Git helpers
git-setup:
	git init
	git add .
	git commit -m "Initial commit"
	@echo "✓ Git repository initialized!"

# Environment info
info:
	@echo "Python version:"
	@python --version
	@echo "\nPip version:"
	@pip --version
	@echo "\nInstalled packages:"
	@pip list | grep -E "torch|numpy|pandas"
	@echo "\nCUDA available:"
	@python -c "import torch; print(torch.cuda.is_available())"
