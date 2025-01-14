from pathlib import Path
import typer
from loguru import logger
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import glob
from lung_cancer_detection.config import MODEL_CHECKPOINTS_DIR, METADATA_DIR, MODELS_DIR, FIGURES_DIR
from lung_cancer_detection.tf_dataset_loader import load_datasets  # Ensure this function is implemented

app = typer.Typer()

def visualize_training(history):
    """Plot training and validation accuracy/loss."""
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

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

@app.command()
def main(
    metadata_path: Path = METADATA_DIR,
    model_checkpoint_path: Path = MODEL_CHECKPOINTS_DIR,
):
    logger.info("Generating Keras Dataset...")
    train_dataset, test_dataset, val_dataset = load_datasets(metadata_path, image_size=(224, 224), batch_size=32)
    logger.success("Features generation complete.")

    # Define input shape for grayscale images
    image_shape = (224, 224, 1)  # Grayscale images have 1 channel
    class_counts = 3  # Update this to match the number of classes in your dataset

    # Define the checkpoint path
    checkpoint_path = str(model_checkpoint_path / "cnn_model_epoch-{epoch:02d}_val_loss-{val_loss:.2f}.keras")
    latest_checkpoint = max(glob.glob(f"{model_checkpoint_path}/*.keras"), default=None, key=lambda x: x)

    # Load model from checkpoint or define a new one
    if latest_checkpoint:
        logger.info(f"Found checkpoint: {latest_checkpoint}. Loading model...")
        cnn_model = load_model(latest_checkpoint)
        initial_epoch = int(latest_checkpoint.split("epoch-")[1].split("_")[0])
    else:
        logger.info("No checkpoint found. Defining model from scratch.")
        initial_epoch = 0

        # Instantiate the CNN Model
        cnn_model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=image_shape, padding='same'),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation='relu', padding='same'),
            MaxPooling2D((2, 2)),
            Conv2D(128, (3, 3), activation='relu', padding='same'),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(class_counts, activation='softmax')
        ])

        cnn_model.summary()

        cnn_model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

    # Define callbacks
    checkpoint_callback = ModelCheckpoint(
        filepath=checkpoint_path,
        monitor='val_loss',
        save_best_only=True,
        save_weights_only=False,
        mode='min',
        verbose=1
    )

    # Train the model
    epochs = 1
    history = cnn_model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        initial_epoch=initial_epoch,
        callbacks=[checkpoint_callback]
    )

    # Visualize training history
    visualize_training(history)


if __name__ == "__main__":
    app()
