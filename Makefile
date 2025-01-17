#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = lung_cancer_detection
PYTHON_VERSION = 3.9.18
PYTHON_INTERPRETER = python

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Install Python Dependencies
## This updates the Conda environment specified by `PROJECT_NAME` using the `environment.yml` file.
## It ensures that dependencies are installed, updated, or removed as needed (`--prune` flag).
.PHONY: requirements
requirements:
	conda env update --name $(PROJECT_NAME) --file environment.yml --prune

## Delete all compiled Python files
## This cleans up the project by removing Python bytecode files (`.pyc`, `.pyo`) and `__pycache__` directories.
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

## Lint using flake8 and black (use `make format` to do formatting)
## Runs code linting tools:
## - `flake8`: Checks for Python code style issues.
## - `isort`: Ensures imports are sorted and formatted according to the `black` profile.
## - `black`: Verifies code formatting based on `pyproject.toml` configuration.
.PHONY: lint
lint:
	flake8 lung_cancer_detection
	isort --check --diff --profile black lung_cancer_detection
	black --check --config pyproject.toml lung_cancer_detection

## Format source code with black
## Automatically reformats the Python code using `black` with the configuration specified in `pyproject.toml`.
.PHONY: format
format:
	black --config pyproject.toml lung_cancer_detection

## Set up python interpreter environment
## Creates a new Conda environment with the name specified by `PROJECT_NAME` using `environment.yml`.
## Provides instructions for activating the environment after creation.
.PHONY: create_environment
create_environment:
	conda env create --name $(PROJECT_NAME) -f environment.yml
	
	@echo ">>> conda env created. Activate with:\nconda activate $(PROJECT_NAME)"

#################################################################################
# PROJECT RULES                                                                 #
#################################################################################

## Make Dataset

## Runs the `dataset.py` script to generate or process the dataset required for training or analysis.
.PHONY: download_data
download_data: requirements
	$(PYTHON_INTERPRETER) lung_cancer_detection/download_dataset.py


## Runs the `dataset.py` script to generate or process the dataset required for training or analysis.
.PHONY: data
data: requirements
	$(PYTHON_INTERPRETER) lung_cancer_detection/dataset.py

## Exploratory Data Analysis
## Executes the `eda.py` script to perform exploratory data analysis on the dataset.
## This typically includes generating visualizations and statistics.
.PHONY: eda
eda: requirements
	$(PYTHON_INTERPRETER) lung_cancer_detection/eda.py

## Train CNN Model
## Trains a Convolutional Neural Network (CNN) model by executing the `train_cnn.py` script.
.PHONY: train_cnn
train_cnn: requirements
	$(PYTHON_INTERPRETER) lung_cancer_detection/train_cnn.py

## Train ViT Model
## Trains a Vision Transformer (ViT) model by executing the `train_vit.py` script.
.PHONY: train_vit
train_vit: requirements
	$(PYTHON_INTERPRETER) lung_cancer_detection/train_vit.py

## Predict CNN Model
## Uses the trained CNN model to make predictions on a test dataset by executing `predict_cnn.py`.
.PHONY: predict_cnn
predict_cnn: requirements
	$(PYTHON_INTERPRETER) lung_cancer_detection/predict_cnn.py

## Predict using ViT Model
## Uses the trained ViT model to make predictions on a test dataset by executing `predict_vit.py`.
.PHONY: predict_vit
predict_vit: requirements
	$(PYTHON_INTERPRETER) lung_cancer_detection/predict_vit.py

#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

## Display help for available commands
## Automatically generates a list of all documented Makefile targets with their descriptions.
help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
