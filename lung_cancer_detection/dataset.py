#!/opt/anaconda3/envs/lung_cancer_detection/bin/python

from pathlib import Path

import typer
from loguru import logger
from tqdm import tqdm
from zipfile import ZipFile
import tensorflow as tf
from keras.preprocessing import image_dataset_from_directory

from lung_cancer_detection.config import UNZIPPED_DATA_DIR, RAW_DATA_DIR, METADATA_DIR

# -------- module to preprocess
import pandas as pd
import os
from sklearn.model_selection import train_test_split
import shutil


app = typer.Typer()


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    input_dir: Path = RAW_DATA_DIR / "raw_data_lung_cancer.zip",
    unzip_dir: Path = UNZIPPED_DATA_DIR,
    output_file: Path = METADATA_DIR,
):
    # ------------------Unzip Dataset ----------------------------

    # ---- Unzip and process dataset ----
    logger.info(f"Unzipping dataset from {input_dir} to {unzip_dir}...")
    with ZipFile(input_dir, "r") as zip:
        zip.extractall(unzip_dir)
        logger.success(f"Dataset successfully unzipped to {unzip_dir}")

    # drop colon image sets

    # Path to the colon_image_sets directory
    colon_dataset_path = UNZIPPED_DATA_DIR / "lung_colon_image_set/colon_image_sets"

    # Check if the directory exists and delete it
    if os.path.exists(colon_dataset_path):
        shutil.rmtree(colon_dataset_path)

    # Move lung_image_sets subfolders up two levels
    lung_image_sets_path = UNZIPPED_DATA_DIR / "lung_colon_image_set/lung_image_sets"
    target_dir = UNZIPPED_DATA_DIR

    if lung_image_sets_path.exists():
        logger.info(f"Moving subfolders from {lung_image_sets_path} to {target_dir}...")
        subfolders = [f for f in lung_image_sets_path.iterdir() if f.is_dir()]
        for subfolder in subfolders:
            shutil.move(str(subfolder), str(target_dir))
            logger.success(f"Moved {subfolder.name} to {target_dir}")

        # Remove now-empty lung_image_sets and lung_colon_image_set directories
        shutil.rmtree(lung_image_sets_path.parent)
        logger.info(f"Deleted empty directories: {lung_image_sets_path.parent}")

    # ------------------create metadata with train / test / splits ----------------------------

    ## Make the metadata file
    label_mapping = {"lung_aca": 0, "lung_n": 1, "lung_scc": 2}

    metadata = []

    for folder_name, label in label_mapping.items():
        folder_path = os.path.join(UNZIPPED_DATA_DIR, folder_name)
        if os.path.exists(folder_path):
            for file_name in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file_name)
                if os.path.isfile(file_path):
                    metadata.append({"file_path": file_path, "label": label})
    metadata_df = pd.DataFrame(metadata)
    metadata_df.to_csv(os.path.join(output_file, "metadata.csv"), index=False)

    ##create a train, test, val split
    train_ds, test_ds = train_test_split(
        metadata_df, test_size=0.3, random_state=42, shuffle=True, stratify=metadata_df["label"]
    )
    test_ds, val_ds = train_test_split(
        test_ds, test_size=(2 / 3), random_state=42, shuffle=True, stratify=test_ds["label"]
    )

    ##export datasets to metadata_folder
    train_ds.to_csv(os.path.join(output_file, "train.csv"), index=False)
    test_ds.to_csv(os.path.join(output_file, "test.csv"), index=False)
    val_ds.to_csv(os.path.join(output_file, "val.csv"), index=False)

    logger.success("Processing Dataset Complete")


if __name__ == "__main__":
    app()
