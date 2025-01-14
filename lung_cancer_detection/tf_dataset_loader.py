import tensorflow as tf                  # For TensorFlow operations
import pandas as pd                      # For reading CSV files
from sklearn.preprocessing import LabelEncoder  # For encoding labels
from pathlib import Path                 # For handling file paths

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

def load_datasets(metadata_dir, image_size=(224, 224), batch_size=32):
    """Load train, test, and validation datasets."""

    def load_dataset(csv_file):
        data = pd.read_csv(csv_file)
        image_paths = data['file_path'].values
        labels = LabelEncoder().fit_transform(data['label'].values)
        dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
        dataset = dataset.shuffle(buffer_size=len(image_paths))
        dataset = dataset.map(lambda x, y: preprocess_image(x, y, image_size), 
                              num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
        return dataset

    train_csv = metadata_dir / "train.csv"
    test_csv = metadata_dir / "test.csv"
    val_csv = metadata_dir / "val.csv"

    train_dataset = load_dataset(train_csv)
    test_dataset = load_dataset(test_csv)
    val_dataset = load_dataset(val_csv)

    return train_dataset, test_dataset, val_dataset
