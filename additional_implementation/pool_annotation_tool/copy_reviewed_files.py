from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile as tiff

ACCEPTED = "accepted"
REJECTED = "rejected"
SELECTED_STATUSES = {ACCEPTED, REJECTED}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Copy accepted and rejected review samples into a new dataset. "
            "Accepted rows receive has_pool=True, rejected rows has_pool=False."
        )
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help=(
            "Source directory containing imgs/, msks/, image_info.csv "
            "and review_status.csv."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Destination directory for the reviewed dataset.",
    )
    parser.add_argument(
        "--review-csv",
        type=Path,
        default=None,
        help="Default: <data-dir>/review_status.csv",
    )
    parser.add_argument(
        "--metadata-csv",
        type=Path,
        default=None,
        help="Default: <data-dir>/image_info.csv",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing images and masks in the output directory.",
    )
    parser.add_argument(
        "--keep-rejected-masks",
        action="store_true",
        help=(
            "Copy the original mask for rejected samples. "
            "Without this option, rejected samples receive an empty mask."
        ),
    )
    return parser.parse_args()


def copy_file(
    source: Path,
    destination: Path,
    overwrite: bool,
) -> bool:
    if destination.exists() and not overwrite:
        return False

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    shutil.copy2(
        source,
        destination,
    )
    return True


def write_empty_mask(
    source_mask: Path,
    destination_mask: Path,
    overwrite: bool,
) -> bool:
    if destination_mask.exists() and not overwrite:
        return False

    mask = tiff.imread(source_mask)

    if mask.ndim == 3:
        empty_mask = np.zeros(
            mask.shape[:2],
            dtype=mask.dtype,
        )
    else:
        empty_mask = np.zeros_like(mask)

    destination_mask.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tiff.imwrite(
        destination_mask,
        empty_mask,
    )
    return True


def main() -> None:
    args = parse_args()

    data_dir = args.data_dir.resolve()
    output_dir = args.output_dir.resolve()

    review_csv = (
        args.review_csv.resolve()
        if args.review_csv is not None
        else data_dir / "review_status.csv"
    )
    metadata_csv = (
        args.metadata_csv.resolve()
        if args.metadata_csv is not None
        else data_dir / "image_info.csv"
    )

    image_dir = data_dir / "imgs"
    mask_dir = data_dir / "msks"

    output_image_dir = output_dir / "imgs"
    output_mask_dir = output_dir / "msks"
    output_metadata_csv = output_dir / "image_info.csv"
    output_review_csv = output_dir / "review_status.csv"

    required_paths = [
        review_csv,
        metadata_csv,
        image_dir,
        mask_dir,
    ]

    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(f"Required path not found: {path}")

    reviews = pd.read_csv(
        review_csv,
        dtype={
            "filename": str,
            "status": str,
            "comment": str,
        },
        keep_default_na=False,
    )

    metadata = pd.read_csv(
        metadata_csv,
        dtype={
            "filename": str,
            "mask_path": str,
        },
        keep_default_na=False,
    )

    missing_review_columns = {
        "filename",
        "status",
    } - set(reviews.columns)

    missing_metadata_columns = {
        "filename",
        "mask_path",
    } - set(metadata.columns)

    if missing_review_columns:
        raise ValueError(
            "Missing columns in review CSV: " f"{sorted(missing_review_columns)}"
        )

    if missing_metadata_columns:
        raise ValueError(
            "Missing columns in metadata CSV: " f"{sorted(missing_metadata_columns)}"
        )

    reviews["status"] = reviews["status"].str.strip().str.lower()

    selected_reviews = reviews[reviews["status"].isin(SELECTED_STATUSES)].copy()

    if selected_reviews.empty:
        print("No accepted or rejected rows were found.")
        return

    selected = selected_reviews.merge(
        metadata,
        on="filename",
        how="left",
        validate="one_to_one",
        suffixes=("_review", ""),
    )

    missing_metadata = selected[
        selected["mask_path"].isna()
        | (selected["mask_path"].astype(str).str.strip() == "")
    ]

    if not missing_metadata.empty:
        print("Warning: reviewed files without matching metadata " "will be skipped:")

        for filename in missing_metadata["filename"]:
            print(f"  - {filename}")

    selected = selected[
        selected["mask_path"].notna()
        & (selected["mask_path"].astype(str).str.strip() != "")
    ].copy()

    output_image_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    output_mask_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    copied_images = 0
    copied_masks = 0
    empty_masks_written = 0
    skipped_existing = 0

    missing_images: list[str] = []
    missing_masks: list[str] = []
    successful_filenames: list[str] = []

    status_by_filename = selected_reviews.set_index("filename")["status"].to_dict()

    for _, row in selected.iterrows():
        filename = str(row["filename"])
        mask_filename = str(row["mask_path"])
        status = status_by_filename[filename]

        source_image = image_dir / filename
        source_mask = mask_dir / mask_filename

        destination_image = output_image_dir / source_image.name
        destination_mask = output_mask_dir / source_mask.name

        if not source_image.exists():
            missing_images.append(str(source_image))
            continue

        if not source_mask.exists():
            missing_masks.append(str(source_mask))
            continue

        image_copied = copy_file(
            source_image,
            destination_image,
            args.overwrite,
        )

        if status == REJECTED and not args.keep_rejected_masks:
            mask_copied = write_empty_mask(
                source_mask,
                destination_mask,
                args.overwrite,
            )

            if mask_copied:
                empty_masks_written += 1
        else:
            mask_copied = copy_file(
                source_mask,
                destination_mask,
                args.overwrite,
            )

            if mask_copied:
                copied_masks += 1

        if image_copied:
            copied_images += 1

        if not image_copied and not mask_copied:
            skipped_existing += 1

        successful_filenames.append(filename)

    successful_set = set(successful_filenames)

    output_metadata = metadata[metadata["filename"].isin(successful_set)].copy()

    output_reviews = reviews[reviews["filename"].isin(successful_set)].copy()

    output_metadata["has_pool"] = (
        output_metadata["filename"].map(status_by_filename).eq(ACCEPTED)
    )

    if "contains_pool" in output_metadata.columns:
        output_metadata["contains_pool"] = output_metadata["has_pool"]

    output_metadata = output_metadata.merge(
        output_reviews[
            [
                column
                for column in [
                    "filename",
                    "status",
                    "comment",
                ]
                if column in output_reviews.columns
            ]
        ],
        on="filename",
        how="left",
        validate="one_to_one",
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_metadata.to_csv(
        output_metadata_csv,
        index=False,
    )
    output_reviews.to_csv(
        output_review_csv,
        index=False,
    )

    accepted_count = int((output_metadata["has_pool"] == True).sum())
    rejected_count = int((output_metadata["has_pool"] == False).sum())

    print()
    print("Finished")
    print("--------")
    print(f"Accepted / has_pool=True: {accepted_count}")
    print(f"Rejected / has_pool=False: {rejected_count}")
    print(f"Complete image-mask pairs: " f"{len(successful_filenames)}")
    print(f"Images copied: {copied_images}")
    print(f"Original masks copied: {copied_masks}")
    print(f"Empty rejected masks written: " f"{empty_masks_written}")
    print(f"Existing pairs skipped: " f"{skipped_existing}")
    print(f"Missing images: {len(missing_images)}")
    print(f"Missing masks: {len(missing_masks)}")
    print(f"Output directory: {output_dir}")
    print(f"Metadata CSV: {output_metadata_csv}")
    print(f"Review CSV: {output_review_csv}")

    if missing_images:
        print()
        print("Missing image files:")

        for path in missing_images:
            print(f"  - {path}")

    if missing_masks:
        print()
        print("Missing mask files:")

        for path in missing_masks:
            print(f"  - {path}")


if __name__ == "__main__":
    main()
