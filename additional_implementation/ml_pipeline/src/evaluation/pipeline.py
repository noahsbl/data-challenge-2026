from __future__ import annotations

import json
import shutil
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
import tifffile as tiff
from PIL import Image
from tqdm import tqdm

from src.data.datasets import to_image_float32
from src.evaluators.cluster_counter_evaluator import ClusterCounterEvaluator
from src.evaluators.confusion_matrix_evaluator import ConfusionMatrixEvaluator
from src.evaluators.iou_evaluator import IntersectionOverUnionEvaluator
from src.evaluators.time_evaluator import TimeEvaluator


def calculate_binary_metrics(
    evaluator: ConfusionMatrixEvaluator,
    image_count: int,
) -> dict[str, float | int]:
    """
    Calculates binary classification metrics from the confusion matrix counters.
    """
    true_positives = evaluator.tp_counter
    false_positives = evaluator.fp_counter
    true_negatives = evaluator.tn_counter
    false_negatives = evaluator.fn_counter

    precision_denominator = true_positives + false_positives
    recall_denominator = true_positives + false_negatives

    precision = (
        true_positives / precision_denominator if precision_denominator > 0 else 0.0
    )
    recall = true_positives / recall_denominator if recall_denominator > 0 else 0.0
    f1_score = (
        2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    )
    accuracy = (
        (true_positives + true_negatives) / image_count if image_count > 0 else 0.0
    )

    return {
        "images": image_count,
        "tp": int(true_positives),
        "fp": int(false_positives),
        "tn": int(true_negatives),
        "fn": int(false_negatives),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1_score),
    }


def image_to_rgb_preview(
    image: np.ndarray,
) -> np.ndarray:
    """
    Converts an input image to an RGB uint8 preview for visualization.
    """
    rgb = to_image_float32(
        image,
        in_channels=3,
    )

    return np.ascontiguousarray((rgb * 255.0).clip(0, 255).astype(np.uint8))


def save_error_artifacts(
    output_directory: Path,
    image: np.ndarray,
    gold_mask: np.ndarray,
    predicted_mask: np.ndarray,
    filename: str,
) -> None:
    """
    Saves the RGB image, ground-truth mask, and predicted mask for an error case.
    """
    output_directory.mkdir(parents=True, exist_ok=True)
    stem = Path(filename).stem

    Image.fromarray(
        image_to_rgb_preview(image),
        mode="RGB",
    ).save(output_directory / f"{stem}_rgb.png")
    Image.fromarray((gold_mask > 0).astype(np.uint8) * 255).save(
        output_directory / f"{stem}_gold.png"
    )
    Image.fromarray((predicted_mask > 0).astype(np.uint8) * 255).save(
        output_directory / f"{stem}_prediction.png"
    )


def evaluate_dataframe(
    classifier,
    dataframe: pd.DataFrame,
    artifact_directory: Path | None = None,
    export_errors: bool = True,
) -> dict[str, Any]:
    """
    Evaluates a classifier on all samples in the given dataframe.

    The function computes binary classification metrics, inference time,
    segmentation metrics for positive samples, and optionally exports false
    positive and false negative examples together with evaluation artifacts.
    """
    confusion_evaluator = ConfusionMatrixEvaluator()
    time_evaluator = TimeEvaluator()
    cluster_evaluator = ClusterCounterEvaluator()
    iou_evaluator = IntersectionOverUnionEvaluator(1.0)

    evaluators = (
        confusion_evaluator,
        time_evaluator,
        cluster_evaluator,
        iou_evaluator,
    )
    for evaluator in evaluators:
        evaluator.reset()

    false_positive_rows: list[dict] = []
    false_negative_rows: list[dict] = []

    if artifact_directory is not None and export_errors:
        for folder_name in ("false_positives", "false_negatives"):
            error_directory = artifact_directory / folder_name
            shutil.rmtree(error_directory, ignore_errors=True)
            error_directory.mkdir(parents=True, exist_ok=True)

    evaluation_start = perf_counter()

    progress = tqdm(
        dataframe.iterrows(),
        total=len(dataframe),
        desc=classifier.__name__,
    )

    for _, row in progress:
        image_path = Path(row["image_root"]) / str(row["filename"])
        mask_path = Path(row["mask_root"]) / str(row["mask_path"])

        image = tiff.imread(image_path)
        gold_mask = tiff.imread(mask_path)
        predicted_mask = classifier.classify(img=image, row=row)

        time_evaluator.evaluate_step(
            msk=predicted_mask,
            gold_msk=gold_mask,
        )
        confusion_evaluator.evaluate_step(
            msk=predicted_mask,
            gold_msk=gold_mask,
        )

        gold_is_positive = bool(np.any(gold_mask))
        prediction_is_positive = bool(np.any(predicted_mask))

        if getattr(classifier, "supports_segmentation", True) and gold_is_positive:
            cluster_evaluator.evaluate_step(
                msk=predicted_mask,
                gold_msk=gold_mask,
            )
            iou_evaluator.evaluate_step(
                msk=predicted_mask,
                gold_msk=gold_mask,
            )

        error_type: str | None = None

        if prediction_is_positive and not gold_is_positive:
            false_positive_rows.append(row.to_dict())
            error_type = "false_positives"
        elif gold_is_positive and not prediction_is_positive:
            false_negative_rows.append(row.to_dict())
            error_type = "false_negatives"

        if error_type is not None and artifact_directory is not None and export_errors:
            save_error_artifacts(
                output_directory=artifact_directory / error_type,
                image=image,
                gold_mask=gold_mask,
                predicted_mask=predicted_mask,
                filename=str(row["filename"]),
            )

    metrics: dict[str, Any] = calculate_binary_metrics(
        evaluator=confusion_evaluator,
        image_count=len(dataframe),
    )
    metrics.update(
        {
            "average_seconds": (
                float(np.mean(time_evaluator.result_times))
                if time_evaluator.result_times
                else 0.0
            ),
            "total_seconds": float(perf_counter() - evaluation_start),
            "cluster_deviation": (
                float(np.mean(cluster_evaluator.results))
                if cluster_evaluator.results
                else 0.0
            ),
            "iou": (
                float(iou_evaluator.intersection_sum / iou_evaluator.union_sum)
                if iou_evaluator.union_sum > 0
                else 1.0
            ),
            "iou_intersection": int(iou_evaluator.intersection_sum),
            "iou_union": int(iou_evaluator.union_sum),
        }
    )

    if artifact_directory is not None:
        artifact_directory.mkdir(parents=True, exist_ok=True)
        (artifact_directory / "metrics.json").write_text(
            json.dumps(metrics, indent=2),
            encoding="utf-8",
        )
        pd.DataFrame(false_positive_rows).to_csv(
            artifact_directory / "false_positives.csv",
            index=False,
        )
        pd.DataFrame(false_negative_rows).to_csv(
            artifact_directory / "false_negatives.csv",
            index=False,
        )

    return metrics
