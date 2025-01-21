# Lung Cancer Detection

[![CCDS Project Template](https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter)](https://cookiecutter-data-science.drivendata.org/)

This project involves the analysis of histopathological images using Convolutional Neural Networks (CNN) and Vision Transformers (ViT) for lung cancer detection. The work compares the performance of these models in identifying lung adenocarcinoma, lung squamous cell carcinoma, and normal lung tissue, leveraging interpretability features such as attention maps.

## Formal Report

A detailed report, *Detecting Lung Cancer in Histopathological Images Using CNN and Visual Transformers*, provides in-depth insights into the project’s methodology, experiments, and findings. This includes a comprehensive comparison of model architectures, performance metrics, and visualizations like attention maps. You can find the report in the `reports` folder under `reports/Detecting_Lung_Cancer_Project_Report.pdf`.

## Project Organization

```plaintext
├── LICENSE            <- MIT
├── Makefile           <- Makefile with convenience commands like `make data`, `make train`, or `make predict`.
├── README.md          <- This README file for developers using and contributing to the project.
├── data
│   ├── external       <- Data from third-party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical datasets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- Documentation files for further details about the project.
│
├── models             <- Trained and serialized models, model predictions, or model summaries.
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata and tools like Black and Flake8.
│
├── references         <- Data dictionaries, manuals, and explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   ├── figures        <- Generated graphics and figures to be used in reporting.
│   └── Detecting_Lung_Cancer_Project_Report.pdf <- The formal project report.
│
├── environment.yml   <- The requirements file for reproducing the analysis environment, e.g.,
│                         generated with `pip freeze > requirements.txt`.
│
├── setup.cfg          <- Configuration file for flake8.
│
└── lung_cancer_detection   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes lung_cancer_detection a Python module.
    │
    ├── config.py               <- Stores useful variables and configuration.
    |
    ├── download_dataset.py     <- Script to download zipped dataset from Kaggle
    │
    ├── dataset.py              <- Scripts to download or generate data.
    │
    ├── features.py             <- Code to create features for modeling.
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models.         
    │   └── train.py            <- Code to train models.
    │
    └── plots.py                <- Code to create visualizations.
```

## Highlights of the Report

The report includes:

1. **Abstract:** A summary of the project, focusing on its objective and key findings.
2. **Introduction:** Context about lung cancer diagnostics and how machine learning can improve detection.
3. **Methodology:** Details on datasets, preprocessing, CNN and ViT architectures, and training configurations.
4. **Results:** Performance metrics (accuracy, precision, recall) and comparisons, including visualizations such as confusion matrices and attention maps.
5. **Discussion:** Insights into model performance, trade-offs, and recommendations for clinical application.
6. **Conclusion:** The transformative potential of AI in lung cancer detection.

For a comprehensive understanding, refer to the [report](reports/Detecting_Lung_Cancer_Project_Report.pdf).

---

## Getting Started

### Prerequisites

- Python 3.9 or higher
- `conda` for environment management

### Setup

The Makefile in this repository simplifies executing key tasks. Below are the commands you can use:

Run the following to create a conda environment and install all dependencies:

```bash
make create_environment
```

#### Generate datasets: 
Use this command to download the dataset from Kaggle:
```bash
make download_data
```


Use this command to preprocess data and generate required datasets:
```bash
make data
```

Run Exploratory Data Analysis (EDA): Perform exploratory analysis with:

```bash
make eda
```

#### Train models:

To train the CNN model:
```bash
make train_cnn
```

To train the Vision Transformer (ViT) model:
```bash
make train_vit
```

Make predictions:
``` bash
make predict_vit
```

Use CNN to predict:
```bash
make predict_cnn
```

Use ViT to predict:
```bash
make predict_vit
```

Check code quality:

Lint the codebase for issues:
```bash
make lint
```

Automatically format the codebase:
```bash
make format
```

See Summary of Make Commands
```bash
make help
```

