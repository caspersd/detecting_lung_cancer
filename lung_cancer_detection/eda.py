from pathlib import Path

import typer
from loguru import logger
from tqdm import tqdm

from lung_cancer_detection.config import METADATA_DIR, FIGURES_DIR
import tensorflow as tf
from tf_dataset_loader import load_datasets
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
from tf_dataset_loader import preprocess_image


app = typer.Typer()


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    input_path: Path = METADATA_DIR,
    output_path: Path = FIGURES_DIR,
    image_size: str = "224,224",
    batch_size: int = 32,
    # -----------------------------------------
):
    # ---- Generate Keras Dataset ----
    logger.info("Generating Keras Dataset...")
    train_dataset, test_dataset, val_dataset = load_datasets(
        METADATA_DIR, image_size=(224, 224), batch_size=32
    )
    logger.success("Features generation complete.")

    # import full dataset
    metadata_path = METADATA_DIR / "metadata.csv"
    full_dataset = pd.read_csv(metadata_path)

    # -------------------------------------------- Generate Pie Chart of Labels --------------------------------------------
    label_counts = Counter(full_dataset["label"])

    # Extract labels and their counts
    label_names = list(label_counts.keys())
    counts = list(label_counts.values())

    # Create the pie chart
    plt.figure(figsize=(8, 6))
    plt.pie(
        counts,
        labels=label_names,
        autopct="%1.1f%%",
        startangle=90,
        colors=plt.cm.tab20c.colors,
        wedgeprops={"alpha": 0.7},
    )
    plt.title("Distribution of Labels in Dataset")
    plt_name = FIGURES_DIR / "Label_Distribution_Pie_Plot"
    plt.savefig(plt_name)

    # -------------------------------------------- Plot Image Before Processing --------------------------------------------
    # Function to extract one sample per class
    def extract_class_samples(full_dataset, num_samples=1):
        class_names = full_dataset["label"].unique()
        samples = []

        for label in class_names:
            subset = full_dataset[full_dataset["label"] == label].head(num_samples)
            samples.append(subset)

        return pd.concat(samples)

    # Extract one example per class
    subset = extract_class_samples(full_dataset, num_samples=1)

    # Calculate grid size (one row per class, two columns for raw and processed)
    num_classes = len(subset["label"].unique())
    grid_rows = num_classes
    grid_cols = 2  # Raw and processed images

    # Create a figure for raw and processed images
    plt.figure(figsize=(10, 5 * num_classes))
    plt.suptitle("Raw and Processed Images (Resized, Greyscaled)")

    for i, (image_path, label) in enumerate(zip(subset["file_path"], subset["label"])):
        try:
            # ---- Load raw image ----
            raw_image = tf.io.read_file(image_path)
            raw_image = tf.image.decode_jpeg(raw_image, channels=3)

            # ---- Preprocess image ----
            preprocessed_image, _ = preprocess_image(image_path, label)

            # ---- Plot raw image ----
            plt.subplot(grid_rows, grid_cols, 2 * i + 1)  # Left column
            plt.imshow(raw_image.numpy())
            plt.title(f"Raw: {label}", fontsize=12)
            plt.axis("off")

            # ---- Plot processed image ----
            plt.subplot(grid_rows, grid_cols, 2 * i + 2)  # Right column
            plt.imshow(preprocessed_image.numpy(), cmap="gray")
            plt.title(f"Processed: {label}", fontsize=12)
            plt.axis("off")

        except Exception as e:
            print(f"Error processing image {image_path}: {e}")

    plt.tight_layout()
    plt_name = FIGURES_DIR / "post_processed_cell_images"
    plt.savefig(plt_name)


if __name__ == "__main__":
    app()
