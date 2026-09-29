from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import numpy as np
import segmentation_models_pytorch as smp
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.datasets import PoolDataset, create_balanced_sampler
from src.data.sources import load_sources
from src.evaluation.classifier_factory import create_classifier
from src.evaluation.pipeline import evaluate_dataframe
from src.evaluation.tensorboard_logger import TensorBoardLogger
from src.evaluation.versioned import run_versioned_evaluation
from src.models.factory import create_model
from src.training.checkpoints import save_checkpoint


def seed_everything(seed: int) -> None:
    """
    Sets random seeds for NumPy and PyTorch to improve reproducibility.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(device_argument: str | None) -> torch.device:
    """
    Resolves the requested compute device.

    If no device is specified, CUDA is used when available and otherwise CPU.
    """
    if device_argument:
        return torch.device(device_argument)

    return torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )


def create_dataloaders(
    config: dict,
    device: torch.device,
) -> tuple[DataLoader, DataLoader, PoolDataset, PoolDataset]:
    """
    Creates training and validation datasets and data loaders.

    The function loads the configured data sources, creates the corresponding
    PoolDataset instances, optionally applies balanced sampling to the training
    set, and configures the data loaders for the selected device.
    """
    training_config = config["training"]
    task = config["task"]

    train_dataframe = load_sources(config, "train")
    validation_dataframe = load_sources(config, "val")

    dataset_task = (
        "classification"
        if task == "classification"
        else "segmentation"
    )

    in_channels = int(
        config.get("model", {}).get(
            "in_channels",
            3,
        )
    )

    train_dataset = PoolDataset(
        dataframe=train_dataframe,
        image_size=training_config["image_size"],
        task=task,
        augment=training_config.get("augment", True),
        in_channels=in_channels,
    )

    validation_dataset = PoolDataset(
        dataframe=validation_dataframe,
        image_size=training_config["image_size"],
        task=task,
        augment=False,
        in_channels=in_channels,
    )

    sampler = None
    if training_config.get("sampler", "balanced") == "balanced":
        sampler = create_balanced_sampler(train_dataframe)

    workers = int(training_config.get("workers", 4))
    common_arguments = {
        "batch_size": int(training_config["batch_size"]),
        "num_workers": workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": workers > 0,
    }

    train_loader = DataLoader(
        train_dataset,
        sampler=sampler,
        shuffle=sampler is None,
        **common_arguments,
    )
    validation_loader = DataLoader(
        validation_dataset,
        shuffle=False,
        **common_arguments,
    )

    print(f"Training images: {len(train_dataframe)}")
    print(train_dataframe["source_name"].value_counts())
    print("Training prevalence:")
    print(train_dataframe["contains_pool"].value_counts())
    print(f"Validation images: {len(validation_dataframe)}")
    print(validation_dataframe["source_name"].value_counts())
    print("Validation prevalence:")
    print(validation_dataframe["contains_pool"].value_counts())

    return (
        train_loader,
        validation_loader,
        train_dataset,
        validation_dataset,
    )


def compute_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    task: str,
    loss_config: dict,
    bce_loss: torch.nn.Module,
    dice_loss: torch.nn.Module,
) -> torch.Tensor:
    """
    Computes the configured training loss for classification or segmentation.

    Classification uses binary cross-entropy. Segmentation combines weighted
    binary cross-entropy and Dice loss.
    """
    if task == "classification":
        return bce_loss(logits, targets)

    return (
        float(loss_config.get("bce_weight", 1.0))
        * bce_loss(logits, targets)
        + float(loss_config.get("dice_weight", 1.0))
        * dice_loss(logits, targets)
    )


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    task: str,
    loss_config: dict,
    bce_loss: torch.nn.Module,
    dice_loss: torch.nn.Module,
    epoch: int,
) -> float:
    """
    Trains the model for one epoch and returns the average training loss.
    """
    model.train()
    total_loss = 0.0
    total_samples = 0

    progress = tqdm(loader, desc=f"epoch {epoch} train", ncols=80)

    for images, targets, _ in progress:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = compute_loss(
            logits=logits,
            targets=targets,
            task=task,
            loss_config=loss_config,
            bce_loss=bce_loss,
            dice_loss=dice_loss,
        )
        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        total_loss += float(loss.item()) * batch_size
        total_samples += batch_size
        progress.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / max(total_samples, 1)


def validate_model(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    task: str,
    loss_config: dict,
    bce_loss: torch.nn.Module,
    dice_loss: torch.nn.Module,
) -> tuple[float, np.ndarray, np.ndarray]:
    """
    Evaluates the model on the validation loader.

    Returns the average validation loss together with predicted probabilities
    and ground-truth targets for threshold optimization.
    """
    model.eval()
    total_loss = 0.0
    total_samples = 0
    probabilities: list[np.ndarray] = []
    gold_targets: list[np.ndarray] = []

    with torch.inference_mode():
        for images, targets, _ in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            logits = model(images)
            loss = compute_loss(
                logits=logits,
                targets=targets,
                task=task,
                loss_config=loss_config,
                bce_loss=bce_loss,
                dice_loss=dice_loss,
            )

            batch_size = images.size(0)
            total_loss += float(loss.item()) * batch_size
            total_samples += batch_size

            probabilities.append(torch.sigmoid(logits).cpu().numpy())
            gold_targets.append(targets.cpu().numpy())

    return (
        total_loss / max(total_samples, 1),
        np.concatenate(probabilities),
        np.concatenate(gold_targets),
    )


def score_threshold(
    probabilities: np.ndarray,
    gold_targets: np.ndarray,
    threshold: float,
    task: str,
) -> dict[str, float]:
    """
    Computes classification and segmentation metrics for a given threshold.
    """
    predicted = probabilities >= threshold
    gold = gold_targets > 0.5

    if task == "classification":
        predicted_images = predicted.reshape(-1)
        gold_images = gold.reshape(-1)
        pixel_iou = 0.0
    else:
        predicted_images = predicted.reshape(len(predicted), -1).any(axis=1)
        gold_images = gold.reshape(len(gold), -1).any(axis=1)

        intersection = np.logical_and(predicted, gold).sum()
        union = np.logical_or(predicted, gold).sum()
        pixel_iou = float(intersection / union) if union else 1.0

    true_positives = int(
        np.logical_and(predicted_images, gold_images).sum()
    )
    false_positives = int(
        np.logical_and(predicted_images, ~gold_images).sum()
    )
    true_negatives = int(
        np.logical_and(~predicted_images, ~gold_images).sum()
    )
    false_negatives = int(
        np.logical_and(~predicted_images, gold_images).sum()
    )

    precision = (
        true_positives / (true_positives + false_positives)
        if true_positives + false_positives
        else 0.0
    )
    recall = (
        true_positives / (true_positives + false_negatives)
        if true_positives + false_negatives
        else 0.0
    )
    f1_score = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    accuracy = (
        (true_positives + true_negatives) / len(gold_images)
        if len(gold_images)
        else 0.0
    )

    return {
        "threshold": float(threshold),
        "f1": float(f1_score),
        "iou": float(pixel_iou),
        "precision": float(precision),
        "recall": float(recall),
        "accuracy": float(accuracy),
        "tp": true_positives,
        "fp": false_positives,
        "tn": true_negatives,
        "fn": false_negatives,
    }


def find_best_threshold(
    probabilities: np.ndarray,
    gold_targets: np.ndarray,
    thresholds: np.ndarray,
    task: str,
    selection_metric: str,
) -> tuple[dict[str, float], list[dict[str, float]]]:
    """
    Evaluates all candidate thresholds and returns the best threshold according
    to the configured selection metric.
    """
    candidates = [
        score_threshold(
            probabilities=probabilities,
            gold_targets=gold_targets,
            threshold=float(threshold),
            task=task,
        )
        for threshold in thresholds
    ]

    best = max(candidates, key=lambda item: item[selection_metric])
    return best, candidates


def create_checkpoint_payload(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    config: dict,
    threshold: float,
    include_optimizer: bool,
) -> dict[str, Any]:
    """
    Creates the checkpoint payload containing model state, training metadata,
    threshold, model configuration, and optionally the optimizer state.
    """
    payload: dict[str, Any] = {
        "model_state_dict": model.state_dict(),
        "epoch": epoch,
        "image_size": config["training"]["image_size"],
        "best_threshold": threshold,
        "model_config": config["model"],
    }

    export_config = getattr(model, "export_architecture_config", None)
    if callable(export_config):
        payload["architecture_config"] = export_config()

    if include_optimizer:
        payload["optimizer_state_dict"] = optimizer.state_dict()

    return payload


def run_pipeline_validation(
    config: dict,
    checkpoint_path: Path,
    threshold: float,
    epoch: int,
    logger: TensorBoardLogger,
) -> dict:
    """
    Runs validation through the benchmark classifier and evaluation pipeline
    using the checkpoint from the current epoch.
    """
    classifier = create_classifier(
        config=config,
        checkpoint=checkpoint_path,
        overrides={"threshold": threshold},
    )
    validation_dataframe = load_sources(config, "val")
    metrics = evaluate_dataframe(
        classifier=classifier,
        dataframe=validation_dataframe,
        artifact_directory=None,
        export_errors=False,
    )
    logger.log_scalars(
        prefix="given_code/val",
        metrics=metrics,
        step=epoch,
    )
    return metrics


def build_thresholds(validation_config: dict) -> np.ndarray:
    """
    Builds the set of thresholds used for validation threshold search.

    Thresholds can either be provided explicitly or generated from a configured
    minimum, maximum, and number of steps.
    """
    search_config = validation_config.get(
        "threshold_search",
        {},
    )

    if "values" in search_config:
        thresholds = np.asarray(
            search_config["values"],
            dtype=np.float32,
        )
    else:
        thresholds = np.linspace(
            float(search_config.get("minimum", 0.1)),
            float(search_config.get("maximum", 0.9)),
            int(search_config.get("steps", 17)),
            dtype=np.float32,
        )

    if thresholds.ndim != 1 or len(thresholds) == 0:
        raise ValueError(
            "threshold_search must contain at least one threshold."
        )

    if np.any((thresholds < 0.0) | (thresholds > 1.0)):
        raise ValueError(
            "All thresholds must be between 0 and 1."
        )

    return np.unique(thresholds)


def train(config: dict) -> Path:
    """
    Runs the complete model training pipeline.

    The function creates data loaders and the model, trains and validates for
    the configured number of epochs, optimizes the decision threshold, stores
    last and best checkpoints, logs metrics to TensorBoard, and optionally
    evaluates the best model on the test split after training.
    """
    training_config = config["training"]
    validation_config = config.get("validation", {})
    optimizer_config = config.get("optimizer", {})
    loss_config = config.get("loss", {})
    task = config["task"]

    seed_everything(int(training_config.get("seed", 42)))
    device = resolve_device(training_config.get("device"))
    print(f"Device: {device}")

    run_directory = Path(config["run_root"]) / config["run_name"]
    run_directory.mkdir(parents=True, exist_ok=True)

    shutil.copy2(
        config["_config_path"],
        run_directory / "config.yaml",
    )

    (
        train_loader,
        validation_loader,
        train_dataset,
        _,
    ) = create_dataloaders(config, device)

    model = create_model(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(optimizer_config.get("learning_rate", 1e-4)),
        weight_decay=float(optimizer_config.get("weight_decay", 1e-4)),
    )

    bce_loss = torch.nn.BCEWithLogitsLoss()
    dice_loss = smp.losses.DiceLoss(
        mode="binary",
        from_logits=True,
    )

    logger = TensorBoardLogger(
        run_directory / "tensorboard" / "training"
    )

    thresholds = build_thresholds(validation_config)
    selection_metric = validation_config.get(
        "selection_metric",
        "iou" if task == "segmentation" else "f1",
    )
    pipeline_every = int(validation_config.get("pipeline_every", 1))
    best_score = float("-inf")

    try:
        for epoch in range(1, int(training_config["epochs"]) + 1):
            train_loss = train_one_epoch(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                device=device,
                task=task,
                loss_config=loss_config,
                bce_loss=bce_loss,
                dice_loss=dice_loss,
                epoch=epoch,
            )

            val_loss, probabilities, gold_targets = validate_model(
                model=model,
                loader=validation_loader,
                device=device,
                task=task,
                loss_config=loss_config,
                bce_loss=bce_loss,
                dice_loss=dice_loss,
            )

            model_metrics, threshold_candidates = find_best_threshold(
                probabilities=probabilities,
                gold_targets=gold_targets,
                thresholds=thresholds,
                task=task,
                selection_metric=selection_metric,
            )

            logger.log_scalars(
                prefix="loss",
                metrics={"train": train_loss, "val": val_loss},
                step=epoch,
            )
            logger.log_scalars(
                prefix="buildin/val",
                metrics=model_metrics,
                step=epoch,
            )

            for candidate in threshold_candidates:
                threshold_key = f"{candidate['threshold']:.2f}"
                logger.log_scalars(
                    prefix=f"threshold_search/val/{threshold_key}",
                    metrics={
                        key: value
                        for key, value in candidate.items()
                        if key != "threshold"
                    },
                    step=epoch,
                )

            last_checkpoint = run_directory / "weights" / "last.pt"
            save_checkpoint(
                last_checkpoint,
                create_checkpoint_payload(
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    config=config,
                    threshold=model_metrics["threshold"],
                    include_optimizer=True,
                ),
            )

            selection_score = model_metrics[selection_metric]
            pipeline_metrics: dict[str, Any] = {}

            if epoch % pipeline_every == 0:
                pipeline_metrics = run_pipeline_validation(
                    config=config,
                    checkpoint_path=last_checkpoint,
                    threshold=model_metrics["threshold"],
                    epoch=epoch,
                    logger=logger,
                )
                selection_score = float(
                    pipeline_metrics.get(selection_metric, selection_score)
                )

            if selection_score > best_score:
                best_score = selection_score
                save_checkpoint(
                    run_directory / "weights" / "best.pt",
                    create_checkpoint_payload(
                        model=model,
                        optimizer=optimizer,
                        epoch=epoch,
                        config=config,
                        threshold=model_metrics["threshold"],
                        include_optimizer=False,
                    ),
                )

            print(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "buildin_val": model_metrics,
                    "given_code_val": pipeline_metrics,
                    "best_selection_score": best_score,
                }
            )

    finally:
        logger.close()

    if config.get("evaluation", {}).get("test_after_training", True):
        best_checkpoint = run_directory / "weights" / "best.pt"
        classifier = create_classifier(config, best_checkpoint)
        test_dataframe = load_sources(config, "test")

        run_versioned_evaluation(
            classifier=classifier,
            config=config,
            dataframe=test_dataframe,
            run_directory=run_directory,
            split="test",
            checkpoint_paths=[best_checkpoint],
            inference=config.get("inference", {}),
        )

    print(f"Training samples per epoch: {len(train_dataset)}")
    return run_directory
