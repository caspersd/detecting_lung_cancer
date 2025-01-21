from pathlib import Path
import typer
from loguru import logger
from transformers import ViTForImageClassification, Trainer, TrainingArguments
from lung_cancer_detection.config import MODEL_CHECKPOINTS_DIR, METADATA_DIR, PREDICTIONS_DIR
from lung_cancer_detection.tf_dataset_loader import (
    load_hf_datasets,
)  # Ensure this function is implemented
import numpy as np
import torch


app = typer.Typer()


@app.command()
def predict(
    input_file: Path = METADATA_DIR / "test.csv",
    metadata_path: Path = METADATA_DIR,
    model_checkpoint_path: Path = MODEL_CHECKPOINTS_DIR / "ViT",
    output_file: Path = PREDICTIONS_DIR / "vit_prediction",
):
    try:
        logger.info("Attempting to load prediction dataset...")
        test_dataset = load_hf_datasets(input_file)
        logger.success("Test dataset loaded successfully")
    except Exception as e:
        logger.warning(
            f"Prediction dataset (defaults to test dataset if none file not specified) failed to load {e}"
        )

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

    # Configure training arguments (important for Trainer initialization, even for prediction)
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

    logger.info("Starting predictions...")
    predictions = trainer.predict(test_dataset)

    logger.success("Predictions completed.")

    # Extract prediction labels
    pred_labels = np.argmax(predictions.predictions, axis=1)

    # Save predictions to file
    with open(output_file, "w") as f:
        f.write("Predicted Labels\n")
        f.writelines(f"{label}\n" for label in pred_labels)

    logger.success(f"Predictions saved to {output_file}")


if __name__ == "__main__":
    app()
