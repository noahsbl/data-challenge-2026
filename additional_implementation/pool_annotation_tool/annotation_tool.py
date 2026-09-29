from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import tifffile as tiff
from PIL import Image


STATUS_UNREVIEWED = "unreviewed"
STATUS_ACCEPTED = "accepted"
STATUS_REJECTED = "rejected"
STATUS_EDIT = "edit"

VALID_STATUSES = {
    STATUS_UNREVIEWED,
    STATUS_ACCEPTED,
    STATUS_REJECTED,
    STATUS_EDIT,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Review RGB images and segmentation masks."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Folder containing imgs/, msks/ and image_info.csv.",
    )
    parser.add_argument(
        "--metadata-csv",
        type=Path,
        default=None,
        help="Default: <data-dir>/image_info.csv",
    )
    parser.add_argument(
        "--review-csv",
        type=Path,
        default=None,
        help="Default: <data-dir>/review_status.csv",
    )
    parser.add_argument(
        "--edit-dir",
        type=Path,
        default=None,
        help="Default: <data-dir>/manual_edit",
    )
    return parser.parse_args()


def to_rgb_uint8(image: np.ndarray) -> np.ndarray:
    """Use the first three channels and convert them to displayable uint8 RGB."""

    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=2)

    if image.ndim != 3 or image.shape[2] < 3:
        raise ValueError(
            f"Expected an image with at least 3 channels, got {image.shape}."
        )

    rgb = image[..., :3]

    if rgb.dtype == np.uint8:
        return np.ascontiguousarray(rgb)

    rgb = rgb.astype(np.float32)

    minimum = rgb.min(axis=(0, 1), keepdims=True)
    maximum = rgb.max(axis=(0, 1), keepdims=True)

    rgb = (rgb - minimum) / np.maximum(maximum - minimum, 1e-8)

    return (rgb * 255).clip(0, 255).astype(np.uint8)


def to_mask_uint8(mask: np.ndarray) -> np.ndarray:
    """Convert the mask to a binary uint8 image."""

    if mask.ndim == 3:
        mask = mask[..., 0]

    return (mask > 0).astype(np.uint8) * 255


def create_overlay(
    rgb: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """Overlay the mask in red on top of the RGB image."""

    overlay = rgb.astype(np.float32).copy()
    mask_binary = mask > 0

    if np.any(mask_binary):
        red = np.zeros_like(overlay)
        red[..., 0] = 255

        overlay[mask_binary] = (
            0.55 * overlay[mask_binary]
            + 0.45 * red[mask_binary]
        )

    return overlay.clip(0, 255).astype(np.uint8)


def read_metadata(path: Path) -> pd.DataFrame:
    dataframe = pd.read_csv(path)

    required_columns = {"filename", "mask_path"}
    missing_columns = required_columns - set(dataframe.columns)

    if missing_columns:
        raise ValueError(
            f"Metadata CSV is missing columns: {sorted(missing_columns)}"
        )

    return dataframe.reset_index(drop=True)


def read_or_create_reviews(
    review_csv: Path,
    metadata: pd.DataFrame,
) -> pd.DataFrame:
    if review_csv.exists():
        reviews = pd.read_csv(review_csv)
    else:
        reviews = pd.DataFrame(
            {
                "filename": metadata["filename"].astype(str),
                "status": STATUS_UNREVIEWED,
                "comment": "",
            }
        )

    if "filename" not in reviews.columns:
        raise ValueError(
            f"{review_csv} has no 'filename' column."
        )

    if "status" not in reviews.columns:
        reviews["status"] = STATUS_UNREVIEWED

    if "comment" not in reviews.columns:
        reviews["comment"] = ""

    reviews["filename"] = reviews["filename"].astype(str)
    reviews["status"] = reviews["status"].fillna(
        STATUS_UNREVIEWED
    ).astype(str)
    reviews["comment"] = reviews["comment"].fillna("").astype(str)

    invalid_statuses = set(reviews["status"]) - VALID_STATUSES
    if invalid_statuses:
        raise ValueError(
            f"Invalid statuses in {review_csv}: {sorted(invalid_statuses)}"
        )

    existing_filenames = set(reviews["filename"])

    missing_rows = metadata[
        ~metadata["filename"].astype(str).isin(existing_filenames)
    ]

    if not missing_rows.empty:
        additions = pd.DataFrame(
            {
                "filename": missing_rows["filename"].astype(str),
                "status": STATUS_UNREVIEWED,
                "comment": "",
            }
        )
        reviews = pd.concat(
            [reviews, additions],
            ignore_index=True,
        )

    valid_filenames = set(metadata["filename"].astype(str))
    reviews = reviews[
        reviews["filename"].isin(valid_filenames)
    ].reset_index(drop=True)

    return reviews


def save_reviews(
    reviews: pd.DataFrame,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    temporary_path = path.with_suffix(".tmp.csv")
    reviews.to_csv(temporary_path, index=False)
    temporary_path.replace(path)


def set_status(
    reviews: pd.DataFrame,
    filename: str,
    status: str,
    comment: str,
) -> pd.DataFrame:
    reviews = reviews.copy()
    reviews["comment"] = reviews["comment"].fillna("").astype(str)

    row_mask = reviews["filename"] == filename

    reviews.loc[row_mask, "status"] = status
    reviews.loc[row_mask, "comment"] = comment

    return reviews


def copy_to_manual_edit(
    row: pd.Series,
    image_path: Path,
    mask_path: Path,
    edit_dir: Path,
) -> None:
    """Copy image, mask and one-row metadata CSV to the manual edit folder."""

    images_dir = edit_dir / "imgs"
    masks_dir = edit_dir / "msks"
    metadata_dir = edit_dir / "metadata"

    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(
        image_path,
        images_dir / image_path.name,
    )
    shutil.copy2(
        mask_path,
        masks_dir / mask_path.name,
    )

    pd.DataFrame([row.to_dict()]).to_csv(
        metadata_dir / f"{image_path.stem}.csv",
        index=False,
    )


def remove_from_manual_edit(
    image_path: Path,
    mask_path: Path,
    edit_dir: Path,
) -> None:
    """Remove files from manual_edit when a sample is reclassified."""

    paths = [
        edit_dir / "imgs" / image_path.name,
        edit_dir / "msks" / mask_path.name,
        edit_dir / "metadata" / f"{image_path.stem}.csv",
    ]

    for path in paths:
        if path.exists():
            path.unlink()


def find_review_index(
    metadata: pd.DataFrame,
    reviews: pd.DataFrame,
    status_filter: str,
) -> list[int]:
    if status_filter == "all":
        return list(range(len(metadata)))

    status_by_filename = reviews.set_index("filename")["status"]

    result = []

    for index, row in metadata.iterrows():
        filename = str(row["filename"])

        if status_by_filename.get(
            filename,
            STATUS_UNREVIEWED,
        ) == status_filter:
            result.append(index)

    return result


def next_unreviewed_index(
    metadata: pd.DataFrame,
    reviews: pd.DataFrame,
    current_index: int,
) -> int:
    status_by_filename = reviews.set_index("filename")["status"]

    total = len(metadata)

    for offset in range(1, total + 1):
        candidate = (current_index + offset) % total
        filename = str(metadata.iloc[candidate]["filename"])

        if status_by_filename.get(
            filename,
            STATUS_UNREVIEWED,
        ) == STATUS_UNREVIEWED:
            return candidate

    return min(current_index + 1, total - 1)


def main() -> None:
    args = parse_args()

    data_dir = args.data_dir.resolve()
    metadata_csv = (
        args.metadata_csv.resolve()
        if args.metadata_csv is not None
        else data_dir / "image_info.csv"
    )
    review_csv = (
        args.review_csv.resolve()
        if args.review_csv is not None
        else data_dir / "review_status.csv"
    )
    edit_dir = (
        args.edit_dir.resolve()
        if args.edit_dir is not None
        else data_dir / "manual_edit"
    )

    image_dir = data_dir / "imgs"
    mask_dir = data_dir / "msks"

    st.set_page_config(
        page_title="Pool Mask Review",
        layout="wide",
    )

    st.title("Pool image and mask review")

    metadata = read_metadata(metadata_csv)
    reviews = read_or_create_reviews(
        review_csv,
        metadata,
    )

    if "current_index" not in st.session_state:
        st.session_state.current_index = 0

    status_counts = reviews["status"].value_counts()

    top_columns = st.columns(4)
    top_columns[0].metric(
        "Accepted",
        int(status_counts.get(STATUS_ACCEPTED, 0)),
    )
    top_columns[1].metric(
        "Rejected",
        int(status_counts.get(STATUS_REJECTED, 0)),
    )
    top_columns[2].metric(
        "Manual edit",
        int(status_counts.get(STATUS_EDIT, 0)),
    )
    top_columns[3].metric(
        "Unreviewed",
        int(status_counts.get(STATUS_UNREVIEWED, 0)),
    )

    filter_value = st.selectbox(
        "Show",
        options=[
            "all",
            STATUS_UNREVIEWED,
            STATUS_ACCEPTED,
            STATUS_REJECTED,
            STATUS_EDIT,
        ],
        index=0,
    )

    visible_indices = find_review_index(
        metadata,
        reviews,
        filter_value,
    )

    if not visible_indices:
        st.info(
            f"No images with status '{filter_value}'."
        )
        return

    current_index = int(
        np.clip(
            st.session_state.current_index,
            0,
            len(metadata) - 1,
        )
    )

    if current_index not in visible_indices:
        current_index = visible_indices[0]
        st.session_state.current_index = current_index

    row = metadata.iloc[current_index]
    filename = str(row["filename"])

    review_row = reviews[
        reviews["filename"] == filename
    ].iloc[0]

    image_path = image_dir / filename
    mask_path = mask_dir / str(row["mask_path"])

    if not image_path.exists():
        st.error(f"Image not found: {image_path}")
        return

    if not mask_path.exists():
        st.error(f"Mask not found: {mask_path}")
        return

    image = tiff.imread(image_path)
    mask = tiff.imread(mask_path)

    rgb = to_rgb_uint8(image)
    binary_mask = to_mask_uint8(mask)
    overlay = create_overlay(rgb, binary_mask)

    st.progress(
        (current_index + 1) / len(metadata),
        text=(
            f"Image {current_index + 1} of {len(metadata)} — "
            f"current status: {review_row['status']}"
        ),
    )

    st.code(str(image_path))

    image_column, rgb_column, mask_column = st.columns(3, gap="medium")
    
    with image_column:
        st.subheader("Overlay")
        st.image(
            overlay,
            width="stretch",
        )
        
    with rgb_column:
        st.subheader("RGB image")
        st.image(
            rgb,
            width="stretch",
        )
        
    with mask_column:
        st.subheader("Mask")
        st.image(
            binary_mask,
            width="stretch"
        )

    action_columns = st.columns([1, 1, 1, 1, 1])

    previous_clicked = action_columns[0].button(
        "← Previous",
        width="stretch",
    )
    accept_clicked = action_columns[1].button(
        "Y — Accept",
        type="primary",
        width="stretch",
    )
    reject_clicked = action_columns[2].button(
        "N — Reject",
        width="stretch",
    )
    edit_clicked = action_columns[3].button(
        "E — Manual edit",
        width="stretch",
    )
    next_clicked = action_columns[4].button(
        "Next →",
        width="stretch",
    )

    comment = st.text_area(
        "Comment",
        value=str(review_row.get("comment", "") or ""),
        key=f"comment_{filename}",
        placeholder=(
            "Optional, for example: second pool is missing."
        ),
    )

    if accept_clicked:
        reviews = set_status(
            reviews,
            filename,
            STATUS_ACCEPTED,
            comment,
        )
        save_reviews(reviews, review_csv)
        remove_from_manual_edit(
            image_path,
            mask_path,
            edit_dir,
        )
        st.session_state.current_index = next_unreviewed_index(
            metadata,
            reviews,
            current_index,
        )
        st.rerun()

    if reject_clicked:
        reviews = set_status(
            reviews,
            filename,
            STATUS_REJECTED,
            comment,
        )
        save_reviews(reviews, review_csv)
        remove_from_manual_edit(
            image_path,
            mask_path,
            edit_dir,
        )
        st.session_state.current_index = next_unreviewed_index(
            metadata,
            reviews,
            current_index,
        )
        st.rerun()

    if edit_clicked:
        reviews = set_status(
            reviews,
            filename,
            STATUS_EDIT,
            comment,
        )
        save_reviews(reviews, review_csv)
        copy_to_manual_edit(
            row,
            image_path,
            mask_path,
            edit_dir,
        )
        st.session_state.current_index = next_unreviewed_index(
            metadata,
            reviews,
            current_index,
        )
        st.rerun()

    if previous_clicked:
        visible_position = visible_indices.index(current_index)
        new_position = max(visible_position - 1, 0)
        st.session_state.current_index = visible_indices[new_position]
        st.rerun()

    if next_clicked:
        visible_position = visible_indices.index(current_index)
        new_position = min(
            visible_position + 1,
            len(visible_indices) - 1,
        )
        st.session_state.current_index = visible_indices[new_position]
        st.rerun()

    st.divider()

    st.write(f"Review CSV: `{review_csv}`")
    st.write(f"Manual edit folder: `{edit_dir}`")


if __name__ == "__main__":
    main()
