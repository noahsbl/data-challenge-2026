from __future__ import annotations

import argparse
import os
import shutil
import sys

import numpy as np
import pandas as pd
import tifffile as tiff
from tqdm import tqdm

# Allow importing classifiers from the sibling ml_pipeline directory.
ML_PIPELINE_DIR = os.path.abspath(os.path.join("..", "ml_pipeline"))
sys.path.insert(0, ML_PIPELINE_DIR)

from src.classifiers.segformer_segmentation_classifier import (
    SegFormerSegmentationClassifier,
)

# Set directories where the image data is located.

INPUT_DIR = os.path.join("..", "data", "unlabeled")
INPUT_IMAGE_DIR = os.path.join(INPUT_DIR, "imgs")
INPUT_CSV_PATH = os.path.join(INPUT_DIR, "image_info.csv")

OUTPUT_DIR = os.path.join("..", "data", "model_labeled")
OUTPUT_IMAGE_DIR = os.path.join(OUTPUT_DIR, "imgs")
OUTPUT_MASK_DIR = os.path.join(OUTPUT_DIR, "msks")
OUTPUT_CSV_PATH = os.path.join(OUTPUT_DIR, "image_info.csv")


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Generate model-labeled pool images from unlabeled data."
    )

    parser.add_argument(
        "--model-path",
        required=True,
        help="Path to the SegFormer checkpoint.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.3,
        help="Segmentation threshold.",
    )
    parser.add_argument(
        "--min-area",
        type=int,
        default=30,
        help="Minimum connected component area in pixels.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device used for inference, e.g. 'cpu' or 'cuda'.",
    )

    return parser.parse_args()


def recreate_output_directories() -> None:
    """Recreate the output directories."""

    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)

    os.makedirs(OUTPUT_IMAGE_DIR, exist_ok=True)
    os.makedirs(OUTPUT_MASK_DIR, exist_ok=True)


def create_mask_filename(image_filename: str) -> str:
    """Create the corresponding mask filename."""

    return f"MSK_{image_filename}"


def main() -> None:
    arguments = parse_arguments()

    classifier = SegFormerSegmentationClassifier(
        model_path=arguments.model_path,
        threshold=arguments.threshold,
        image_size=512,
        device=arguments.device,
        min_area=arguments.min_area,
    )

    recreate_output_directories()

    dataframe = pd.read_csv(INPUT_CSV_PATH)
    positive_rows: list[pd.Series] = []

    print("Starting classification...")

    for _, original_row in tqdm(
        dataframe.iterrows(),
        total=len(dataframe),
    ):
        image_filename = str(original_row["filename"])

        image_path = os.path.join(
            INPUT_IMAGE_DIR,
            image_filename,
        )

        if not os.path.exists(image_path):
            print(f"Skipping missing image: {image_path}")
            continue

        image = tiff.imread(image_path)

        predicted_mask = classifier.classify(
            img=image,
            row=original_row,
        )

        if not np.any(predicted_mask):
            continue

        mask_filename = create_mask_filename(image_filename)

        output_row = original_row.copy()
        output_row["mask_path"] = mask_filename
        output_row["contains_pool"] = True
        output_row["split"] = "train"

        positive_rows.append(output_row)

        shutil.copy2(
            image_path,
            os.path.join(
                OUTPUT_IMAGE_DIR,
                image_filename,
            ),
        )

        tiff.imwrite(
            os.path.join(
                OUTPUT_MASK_DIR,
                mask_filename,
            ),
            predicted_mask.astype(np.uint8),
        )

    output_columns = list(dataframe.columns)

    if "split" not in output_columns:
        output_columns.append("split")

    if positive_rows:
        output_dataframe = pd.DataFrame(positive_rows)
        output_dataframe = output_dataframe.reindex(columns=output_columns)
    else:
        output_dataframe = pd.DataFrame(columns=output_columns)

    output_dataframe.to_csv(
        OUTPUT_CSV_PATH,
        index=False,
    )

    print()
    print("========================================")
    print(f"Input images: {len(dataframe)}")
    print(f"Positive images: {len(output_dataframe)}")
    print(f"Images saved to: {OUTPUT_IMAGE_DIR}")
    print(f"Masks saved to: {OUTPUT_MASK_DIR}")
    print(f"CSV saved to: {OUTPUT_CSV_PATH}")


if __name__ == "__main__":
    main()
