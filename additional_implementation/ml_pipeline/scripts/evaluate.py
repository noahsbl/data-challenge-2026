from __future__ import annotations

import argparse
from pathlib import Path

from src.data.sources import load_sources
from src.evaluation.classifier_factory import create_classifier
from src.evaluation.versioned import run_versioned_evaluation
from src.utils.config import load_config


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the evaluation script."""
    parser = argparse.ArgumentParser(
        description="Run a versioned evaluation for an existing model run."
    )
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--min-area", type=int)
    parser.add_argument("--classification-threshold", type=float)
    parser.add_argument("--segmentation-threshold", type=float)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def build_inference_overrides(args: argparse.Namespace) -> dict:
    """Build inference overrides from explicitly provided command-line arguments."""
    values = {
        "threshold": args.threshold,
        "min_area": args.min_area,
        "classification_threshold": args.classification_threshold,
        "segmentation_threshold": args.segmentation_threshold,
    }
    return {key: value for key, value in values.items() if value is not None}


def resolve_checkpoints(config: dict, run_directory: Path) -> list[Path]:
    """Resolve the checkpoint files required for the configured model type."""
    model_type = config["model"]["type"]

    if model_type in {"resnet18_classifier", "unet", "segformer"}:
        return [run_directory / "weights" / "best.pt"]

    raise ValueError(f"Unsupported model type: {model_type}")


def main() -> None:
    """
    Run the versioned evaluation.

    Loads the run configuration, creates the classifier, evaluates the selected
    dataset split, stores the evaluation results, and prints the resulting metrics.
    """
    args = parse_args()
    config = load_config(args.run / "config.yaml")
    overrides = build_inference_overrides(args)
    checkpoints = resolve_checkpoints(config, args.run)

    primary_checkpoint = (
        checkpoints[0]
        if config["model"]["type"] in {"resnet18_classifier", "unet", "segformer"}
        else None
    )

    classifier = create_classifier(
        config=config,
        checkpoint=primary_checkpoint,
        overrides=overrides,
    )

    dataframe = load_sources(config, args.split)
    inference = {**config.get("inference", {}), **overrides}

    metrics, evaluation_id = run_versioned_evaluation(
        classifier=classifier,
        config=config,
        dataframe=dataframe,
        run_directory=args.run,
        split=args.split,
        checkpoint_paths=checkpoints,
        inference=inference,
        force=args.force,
    )

    print(f"Evaluation ID: {evaluation_id}")
    print(metrics)


if __name__ == "__main__":
    main()
