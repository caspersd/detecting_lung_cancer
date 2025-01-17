import tensorflow as tf  # For TensorFlow operations
import pandas as pd  # For reading CSV files
from sklearn.preprocessing import LabelEncoder  # For encoding labels
from pathlib import Path  # For handling file paths


def preprocess_image(image_path, label, image_size=(224, 224)):
    """Preprocess a single image: read, convert to grayscale, resize, and normalize."""
    # Load and decode the image
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)  # Load as RGB

    # Convert to grayscale
    image = tf.image.rgb_to_grayscale(image)  # Converts to 1 channel

    # Resize and normalize pixel values
    image = tf.image.resize(image, image_size) / 255.0

    return image, label


def load_datasets(file_path, image_size=(224, 224), batch_size=32):
    """Load train, test, and validation datasets."""

    def load_dataset(csv_file):
        data = pd.read_csv(csv_file)
        image_paths = data["file_path"].values
        labels = LabelEncoder().fit_transform(data["label"].values)
        dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
        dataset = dataset.shuffle(buffer_size=len(image_paths))
        dataset = dataset.map(
            lambda x, y: preprocess_image(x, y, image_size), num_parallel_calls=tf.data.AUTOTUNE
        )
        dataset = dataset.batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
        return dataset

    processed_dataset = load_dataset(file_path)

    return processed_dataset


from datasets import load_dataset, Image
from transformers import ViTImageProcessorFast
from pathlib import Path


def load_hf_datasets(filename):
    # Load the image processor
    processor = ViTImageProcessorFast.from_pretrained("google/vit-base-patch16-224")

    # Row-wise preprocessing function
    def vis_preprocess_image(batch):
        # Convert each image in the batch to grayscale and then back to RGB
        processed_images = []
        for img in batch["file_path"]:
            if isinstance(img, list):  # If images are nested in a list, get the first item
                img = img[0]
            img = img.convert("L").convert("RGB")  # Convert to grayscale
            processed_images.append(img)

        # Process the entire batch of images with the processor
        pixel_values = processor(images=processed_images, return_tensors="pt")["pixel_values"]
        # Combine processed pixel_values with labels
        processed_batch = {
            "pixel_values": pixel_values,
            "labels": batch["label"],
        }

        return processed_batch

    # Function to load and preprocess a dataset split
    def load_dataset_split(csv_file):
        # Load the dataset
        dataset = load_dataset("csv", data_files=str(csv_file))

        # Cast the `file_path` column to Image
        dataset = dataset.cast_column("file_path", Image())

        # Apply row-wise transformations
        dataset = dataset.with_transform(
            vis_preprocess_image,
        )
        print(f"Dataset: {csv_file}:\n {dataset['train']}")
        return dataset["train"]

    # Load and preprocess datasets
    dataset = load_dataset_split(filename)

    return dataset
