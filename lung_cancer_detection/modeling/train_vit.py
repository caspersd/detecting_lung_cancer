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
    FIGURES_DIR,
    MODELS_DIR,
)
from lung_cancer_detection.tf_dataset_loader import load_hf_datasets
from transformers import ViTForImageClassification, TrainingArguments, Trainer
from evaluate import load
import os
import numpy as np
from PIL import Image


app = typer.Typer()


def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    classes = ["aca", "normal", "scc"]
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix for Transformer Model")
    plt_name = FIGURES_DIR / "Conv_NN_Confusion_Matrix.png"
    plt.savefig(plt_name)


def generate_attention_images(model, dataset, classes, output_dir):
    """Generate attention image examples for each label."""
    model.eval()
    model.to("cuda")

    for label in range(len(classes)):
        # Get an example for the target label
        example = next(item for item in dataset if item["labels"] == label)

        pixel_values = example["pixel_values"].unsqueeze(0).to("cuda")  # Add batch dimension
        processed_image = (
            example["pixel_values"].permute(1, 2, 0).cpu().numpy()
        )  # Convert to HWC format for plotting
        processed_image = (processed_image - processed_image.min()) / (
            processed_image.max() - processed_image.min()
        )  # Normalize to [0, 1]
        image_label = example["labels"]

        # Pass through the model to get attention
        with torch.no_grad():
            outputs = model(pixel_values, output_attentions=True)

        attention_weights = outputs.attentions  # List of attention matrices

        # Process the attention map from the last layer
        attention_map = attention_weights[-1][0].mean(dim=0).cpu().numpy()
        cls_attention = attention_map[0, 1:]  # Exclude CLS token self-attention
        cls_attention = cls_attention / cls_attention.max()  # Normalize to [0, 1]

        # Reshape to a grid
        num_patches = int(np.sqrt(cls_attention.shape[0]))
        cls_attention = cls_attention.reshape(num_patches, num_patches)

        # Resize attention map to match the processed image size
        attention_map_resized = np.array(
            Image.fromarray((cls_attention * 255).astype(np.uint8)).resize(
                (processed_image.shape[1], processed_image.shape[0]), resample=Image.BILINEAR
            )
        )

        # Plot the processed image and attention map
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(processed_image)
        plt.title(f"Processed Image: {classes[label]}")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.imshow(processed_image)
        plt.imshow(attention_map_resized, cmap="jet", alpha=0.6)
        plt.title(f"Attention Map: {classes[label]}")
        plt.axis("off")

        plt.tight_layout()
        attention_image_path = output_dir / f"attention_example_label_{label}.png"
        plt.savefig(attention_image_path)
        plt.close()


@app.command()
def main(
    metadata_path: Path = METADATA_DIR,
    model_checkpoint_path: Path = MODEL_CHECKPOINTS_DIR / "ViT",
    Model_dir: path = MODELS_DIR,
):
    logger.info("Generating Hugging Face Dataset...")
    train_path = metadata_path / "train.csv"
    val_path = metadata_path / "val.csv"
    test_path = metadata_path / "test.csv"
    train_dataset = load_hf_datasets(train_path)
    test_dataset = load_hf_datasets(test_path)
    val_dataset = load_hf_datasets(val_path)
    logger.success("Features generation complete.")

    logger.info("loading mdoel ...")
    checkpoints = [
        f.path for f in os.scandir(model_checkpoint_path) if f.is_dir() and "checkpoint" in f.name
    ]
    latest_checkpoint = max(checkpoints, default=None)
    if latest_checkpoint:
        model = ViTForImageClassification.from_pretrained(latest_checkpoint)
        logger.success("Model successfully loaded from checkpoint.")

    else:
        logger.info("No checkpoint found, loading model with initial weights.")

        model_name = "google/vit-base-patch16-224-in21k"
        model = ViTForImageClassification.from_pretrained(
            model_name,
            num_labels=3,  # Adjust based on your dataset
            id2label={0: "lung_aca", 1: "lung_n", 2: "lung_scc"},
            label2id={"lung_aca": 0, "lung_n": 1, "lung_scc": 2},
        )
        logger.success("Model successfully loaded with initial weights.")

    training_args = TrainingArguments(
        output_dir=model_checkpoint_path,
        eval_strategy="steps",
        eval_steps=20,
        save_steps=20,
        learning_rate=2e-4,
        remove_unused_columns=False,
        push_to_hub=False,
        num_train_epochs=5,
        per_device_train_batch_size=400,
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        fp16=True,  # Change this to False if training on CPU
        dataloader_num_workers=os.cpu_count(),
        dataloader_pin_memory=True,
        report_to="none",
    )

    metric = load("accuracy")

    def compute_metrics(p):
        predictions = np.argmax(p.predictions, axis=1)
        return metric.compute(predictions=predictions, references=p.label_ids)

    def collate_fn(batch):
        return {
            "pixel_values": torch.stack([x["pixel_values"] for x in batch]),
            "labels": torch.tensor([x["labels"] for x in batch]),
        }

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        data_collator=collate_fn,
    )

    if latest_checkpoint:
        trainer.train(resume_from_checkpoint=latest_checkpoint)
    else:
        trainer.train()

    Model_dir = Model_dir / f"vit_checkpoint"

    best_metric_name = "accuracy"
    best_metric_value = trainer.state.best_metric  # The value of the best metric
    model_name = f"vit_model_best_{best_metric_name}_{best_metric_value:.4f}"

    # Save the best model
    best_model_dir = Model_dir / model_name
    trainer.save_model(best_model_dir)

    metrics = trainer.evaluate(test_dataset)

    # Collect metrics from log history
    log_history = trainer.state.log_history

    epochs = []
    eval_losses = []
    eval_accuracies = []
    for log in log_history:
        if "eval_loss" in log:
            epochs.append(log["epoch"])
            eval_losses.append(log["eval_loss"])
            eval_accuracies.append(log["eval_accuracy"])

    # Plot metrics across epochs
    plt.plot(epochs, eval_losses, label="Loss")
    plt.plot(epochs, eval_accuracies, label="Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Metrics")
    plt.legend()
    plt.savefig(FIGURES_DIR / "ViT_Loss_Accuracy_Plot.png")

    logger.info("Predicting on Test Dataset")
    predictions = trainer.predict(test_dataset)

    logger.success("Predictions completed.")
    logger.info("Generating Confusion Matrix")
    # Extract prediction labels
    pred_labels = np.argmax(predictions.predictions, axis=1)

    # Define the mapping for string labels to numbers
    label_mapping = {"aca": 0, "normal": 1, "scc": 2}

    # Extract labels and map them to numbers
    true_labels = [label_mapping[example["labels"]] for example in test_dataset]

    # Convert to a NumPy array
    true_labels = np.array(labels)
    plot_confusion_matrix(true_labels, pred_labels)

    classes = ["aca", "normal", "scc"]

    generate_attention_images(model, train_dataset, classes, FIGURES_DIR)


if __name__ == "__main__":
    app()
