from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.evaluation.pipeline import evaluate_dataframe
from src.evaluation.tensorboard_logger import TensorBoardLogger
from src.utils.hashing import file_sha256, stable_hash


def build_evaluation_payload(
    dataframe: pd.DataFrame,
    split: str,
    checkpoint_paths: list[Path],
    inference: dict,
) -> dict:
    """
    Builds a stable evaluation payload containing split, inference settings,
    dataset sources, and checkpoint hashes.
    """
    return {
        "split": split,
        "inference": inference,
        "sources": sorted(dataframe["source_name"].unique().tolist()),
        "source_csvs": sorted(dataframe["source_csv"].unique().tolist()),
        "checkpoints": [
            {
                "path": str(path.resolve()),
                "sha256": file_sha256(path),
            }
            for path in checkpoint_paths
        ],
    }


def update_evaluation_index(
    index_path: Path,
    evaluation_id: str,
    split: str,
    inference: dict,
    metrics: dict,
) -> None:
    """
    Updates the evaluation index with the metrics and settings of one evaluation.
    """
    row = {
        "evaluation_id": evaluation_id,
        "split": split,
        **inference,
        **metrics,
    }

    if index_path.exists():
        index = pd.read_csv(index_path)
        if "evaluation_id" in index.columns:
            index = index[index["evaluation_id"] != evaluation_id]
    else:
        index = pd.DataFrame()

    updated_index = pd.concat(
        [index, pd.DataFrame([row])],
        ignore_index=True,
    )
    updated_index.to_csv(index_path, index=False)


def run_versioned_evaluation(
    classifier,
    config: dict,
    dataframe: pd.DataFrame,
    run_directory: Path,
    split: str,
    checkpoint_paths: list[Path],
    inference: dict,
    force: bool = False,
) -> tuple[dict, str]:
    """
    Runs a versioned evaluation for the given classifier and dataset split.

    The function creates a reproducible evaluation ID from the configuration,
    dataset sources, and checkpoints, reuses existing results unless forced,
    stores metrics and settings, logs results to TensorBoard, and optionally
    evaluates metrics separately for each data source.
    """
    payload = build_evaluation_payload(
        dataframe=dataframe,
        split=split,
        checkpoint_paths=checkpoint_paths,
        inference=inference,
    )

    evaluation_id = f"eval_{stable_hash(payload)}"
    evaluation_directory = run_directory / "evaluations" / evaluation_id
    metrics_path = evaluation_directory / "metrics.json"

    if metrics_path.exists() and not force:
        print(f"Evaluation already exists: {evaluation_id}")
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        return metrics, evaluation_id

    evaluation_directory.mkdir(parents=True, exist_ok=True)

    metrics = evaluate_dataframe(
        classifier=classifier,
        dataframe=dataframe,
        artifact_directory=evaluation_directory,
        export_errors=config.get("evaluation", {}).get(
            "export_errors",
            True,
        ),
    )

    settings = {
        **payload,
        "evaluation_id": evaluation_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    (evaluation_directory / "settings.json").write_text(
        json.dumps(settings, indent=2),
        encoding="utf-8",
    )

    logger = TensorBoardLogger(evaluation_directory / "tensorboard")
    logger.log_scalars(
        prefix=f"pipeline/{split}/overall",
        metrics=metrics,
        step=0,
    )

    if config.get("evaluation", {}).get("report_per_source", True):
        for source_name, source_dataframe in dataframe.groupby("source_name"):
            source_metrics = evaluate_dataframe(
                classifier=classifier,
                dataframe=source_dataframe,
                artifact_directory=(
                    evaluation_directory / f"source_{source_name}"
                ),
                export_errors=False,
            )
            logger.log_scalars(
                prefix=f"pipeline/{split}/source_{source_name}",
                metrics=source_metrics,
                step=0,
            )

    logger.close()

    update_evaluation_index(
        index_path=run_directory / "evaluations" / "index.csv",
        evaluation_id=evaluation_id,
        split=split,
        inference=inference,
        metrics=metrics,
    )

    return metrics, evaluation_id
