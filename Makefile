.PHONY: help install test lint format clean setup

help: ## Show this help message
	@echo "Video Transcript QA Dataset Generation Framework"
	@echo "================================================"
	@echo ""
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

setup: ## Set up the development environment
	@echo "Setting up development environment..."
	conda create --prefix ./envs/trqa python=3.10.0 --no-deps --channel conda-forge --channel defaults --override-channels -y
	@echo "Environment created. Activate with: conda activate ./envs/trqa"

install: ## Install dependencies
	@echo "Installing dependencies..."
	pip install -r requirements.txt
	pip install -e .

install-dev: ## Install development dependencies
	@echo "Installing development dependencies..."
	pip install -r requirements.txt
	pip install -e ".[dev]"

test: ## Run tests
	@echo "Running tests..."
	pytest tests/ -v

test-coverage: ## Run tests with coverage
	@echo "Running tests with coverage..."
	pytest tests/ --cov=src --cov-report=html --cov-report=term

lint: ## Run linting checks
	@echo "Running linting checks..."
	flake8 src/ tests/
	black --check src/ tests/
	isort --check-only src/ tests/

format: ## Format code
	@echo "Formatting code..."
	black src/ tests/
	isort src/ tests/

clean: ## Clean up generated files
	@echo "Cleaning up..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/
	rm -rf dist/
	rm -rf htmlcov/
	rm -rf .pytest_cache/

build: ## Build the package
	@echo "Building package..."
	python setup.py sdist bdist_wheel

publish: ## Publish to PyPI (requires twine)
	@echo "Publishing to PyPI..."
	twine upload dist/*

example: ## Run the example script
	@echo "Running example..."
	python examples/example_usage.py

cli-help: ## Show CLI help
	@echo "CLI Help:"
	python -m src.cli --help

create-config: ## Create a configuration template
	@echo "Creating configuration template..."
	python -m src.cli create-config-template

# Development workflow
dev-setup: setup install-dev ## Complete development setup
	@echo "Development environment ready!"

dev-test: format lint test ## Run full development checks
	@echo "All checks passed!"

# Docker commands (if needed)
docker-build: ## Build Docker image
	docker build -t video-qa-framework .

docker-run: ## Run Docker container
	docker run -it --rm video-qa-framework

# Environment management
env-activate: ## Show how to activate environment
	@echo "To activate the environment, run:"
	@echo "conda activate ./envs/trqa"

env-deactivate: ## Show how to deactivate environment
	@echo "To deactivate the environment, run:"
	@echo "conda deactivate"

# Documentation
docs: ## Generate documentation
	@echo "Generating documentation..."
	pdoc --html src/ --output-dir docs/

serve-docs: ## Serve documentation locally
	@echo "Serving documentation at http://localhost:8000"
	python -m http.server 8000 -d docs/

# Quick start commands
quick-start: ## Quick start guide
	@echo "Quick Start Guide:"
	@echo "1. make setup"
	@echo "2. conda activate ./envs/trqa"
	@echo "3. make install"
	@echo "4. export OPENAI_API_KEY='your-key-here'"
	@echo "5. python -m src.cli generate --help"

# Utility commands
check-deps: ## Check for outdated dependencies
	@echo "Checking for outdated dependencies..."
	pip list --outdated

update-deps: ## Update dependencies
	@echo "Updating dependencies..."
	pip install --upgrade -r requirements.txt

# Git hooks (if using pre-commit)
install-hooks: ## Install git hooks
	@echo "Installing git hooks..."
	pre-commit install

run-hooks: ## Run git hooks
	@echo "Running git hooks..."
	pre-commit run --all-files
