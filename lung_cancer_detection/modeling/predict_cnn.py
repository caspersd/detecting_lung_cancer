from pathlib import Path
import typer
from loguru import logger
from keras.models import load_model
import numpy as np
from lung_cancer_detection.config import METADATA_DIR, MODELS_DIR, PREDICTIONS_DIR
from lung_cancer_detection.tf_dataset_loader import (
    load_datasets,
)  # Ensure this function is implemented
import re

app = typer.Typer()


@app.command()
def predict(
    metadata_path: Path = METADATA_DIR,
    models_dir: Path = MODELS_DIR,
    output_file: Path = PREDICTIONS_DIR / "cnn_predictions.txt",
):
    # Ensure the output directory exists
    output_file.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Scanning for the best model in the models directory...")
    # Find the best model file based on accuracy in the file name
    model_files = list(models_dir.glob("cnn_model_best_accuracy_*.keras"))
    if not model_files:
        logger.error("No models found in the specified models directory.")
        return

    # Extract accuracy from file names and find the highest accuracy
    best_model_file = None
    best_accuracy = -1
    accuracy_pattern = re.compile(r"cnn_model_best_accuracy_(\d+\.\d+).keras")

    for model_file in model_files:
        match = accuracy_pattern.search(model_file.name)
        if match:
            accuracy = float(match.group(1))
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model_file = model_file

    if not best_model_file:
        logger.error("Could not find a valid model file with accuracy in the name.")
        return

    logger.info(f"Best model found: {best_model_file} with accuracy: {best_accuracy:.4f}")

    logger.info("Loading the test dataset for predictions...")
    # Load the test dataset
    test_path = metadata_path / "test.csv"
    test_dataset = load_datasets(test_path, image_size=(224, 224), batch_size=32)
    logger.success("Test dataset loaded successfully.")

    logger.info(f"Loading the best model from: {best_model_file}")
    # Load the best model
    cnn_model = load_model(best_model_file)
    logger.success("Best model loaded successfully.")

    logger.info("Starting predictions...")
    # Perform predictions
    predictions = cnn_model.predict(test_dataset)

    # Get predicted class labels
    pred_labels = np.argmax(predictions, axis=1)

    # Save predictions to a file
    with open(output_file, "w") as f:
        f.write("Predicted Labels\n")
        for label in pred_labels:
            f.write(f"{label}\n")

    logger.success(f"Predictions saved to {output_file}")


if __name__ == "__main__":
    app()
