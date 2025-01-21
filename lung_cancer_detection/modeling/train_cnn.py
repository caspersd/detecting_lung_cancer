from pathlib import Path
import typer
from loguru import logger
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import glob
from lung_cancer_detection.config import (
    MODEL_CHECKPOINTS_DIR,
    METADATA_DIR,
    MODELS_DIR,
    FIGURES_DIR,
)
from lung_cancer_detection.tf_dataset_loader import (
    load_datasets,
)  # Ensure this function is implemented
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np

app = typer.Typer()


def visualize_training(history):
    """Plot training and validation accuracy/loss."""
    acc = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]

    epochs_range = range(len(acc))

    plt.figure(figsize=(12, 6))
    plt.suptitle("Training and Validation Metrics", fontsize=16, y=1.02)

    # Accuracy Plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label="Training Accuracy")
    plt.plot(epochs_range, val_acc, label="Validation Accuracy")
    plt.legend(loc="lower right")
    plt.title("Training and Validation Accuracy")

    # Loss Plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label="Training Loss")
    plt.plot(epochs_range, val_loss, label="Validation Loss")
    plt.legend(loc="upper right")
    plt.title("Training and Validation Loss")

    # Save the figure before showing it
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)  # Adjust space for the suptitle
    plt_name = FIGURES_DIR / "Conv_NN_train_stats.png"
    plt.savefig(plt_name)
    plt.show()


def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    classes = ["aca", "normal", "scc"]
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix for CNN Model")
    plt_name = FIGURES_DIR / "Conv_NN_Confusion_Matrix.png"
    plt.savefig(plt_name)


@app.command()
def main(
    metadata_path: Path = METADATA_DIR,
    model_checkpoint_path: Path = MODEL_CHECKPOINTS_DIR / "CNN",
):
    logger.info("Generating Keras Dataset...")
    train_path = metadata_path / "train.csv"
    test_path = metadata_path / "train.csv"
    val_path = metadata_path / "train.csv"
    train_dataset = load_datasets(train_path, image_size=(224, 224), batch_size=32)
    val_dataset = load_datasets(val_path, image_size=(224, 224), batch_size=32)
    test_dataset = load_datasets(test_path, image_size=(224, 224), batch_size=32)
    logger.success("Features generation complete.")

    # Define input shape for grayscale images
    image_shape = (224, 224, 1)  # Grayscale images have 1 channel
    class_counts = 3  # Update this to match the number of classes in your dataset

    # Define the checkpoint path
    checkpoint_path = str(
        model_checkpoint_path / "cnn_model_epoch-{epoch:02d}_val_loss-{val_loss:.2f}.keras"
    )
    latest_checkpoint = max(
        glob.glob(f"{model_checkpoint_path}/*.keras"), default=None, key=lambda x: x
    )

    # Load model from checkpoint or define a new one
    if latest_checkpoint:
        logger.info(f"Found checkpoint: {latest_checkpoint}. Loading model...")
        cnn_model = load_model(latest_checkpoint)
        initial_epoch = int(latest_checkpoint.split("epoch-")[1].split("_")[0])
        logger.success("Model loaded from checkpoint . . .")
    else:
        logger.info("No checkpoint found. Defining model from scratch.")
        initial_epoch = 0

        # Instantiate the CNN Model
        cnn_model = Sequential(
            [
                Conv2D(32, (3, 3), activation="relu", input_shape=image_shape, padding="same"),
                MaxPooling2D((2, 2)),
                Conv2D(64, (3, 3), activation="relu", padding="same"),
                MaxPooling2D((2, 2)),
                Conv2D(128, (3, 3), activation="relu", padding="same"),
                MaxPooling2D((2, 2)),
                Flatten(),
                Dense(128, activation="relu"),
                Dropout(0.5),
                Dense(class_counts, activation="softmax"),
            ]
        )

        cnn_model.summary()

        cnn_model.compile(
            optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
        )

    # Define callbacks
    checkpoint_callback = ModelCheckpoint(
        filepath=checkpoint_path,
        monitor="val_accuracy",
        save_best_only=False,
        save_weights_only=False,
        mode="max",
        verbose=1,
    )
    best_model_path = str(MODELS_DIR / "cnn_model_best_accuracy_{val_accuracy:.4f}.keras")

    best_model_callback = ModelCheckpoint(
        filepath=best_model_path,
        monitor="val_accuracy",
        save_best_only=True,
        save_weights_only=False,
        mode="max",
        verbose=1,
    )

    # Train the model
    epochs = 5
    history = cnn_model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        initial_epoch=initial_epoch,
        callbacks=[checkpoint_callback, best_model_callback],
    )
    if initial_epoch < epochs:
        # Visualize training history
        visualize_training(history)

    # Perform predictions
    y_pred = cnn_model.predict(test_dataset)

    # Get predicted class labels
    y_pred = np.argmax(y_pred, axis=1)
    true_labels = np.concatenate([label.numpy() for _, label in test_dataset])

    plot_confusion_matrix(true_labels, y_pred)


if __name__ == "__main__":
    app()
