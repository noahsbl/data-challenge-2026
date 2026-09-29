# ML Pipeline

This directory contains the training and evaluation pipeline used for the implemented pool detection and segmentation models, including ResNet, U-Net and SegFormer.

## Installation

From this directory, install the required packages with:

```bash
pip install -r requirements.txt
```

## Structure

```text
ml_pipeline/
├── configs/        # model, data, training and inference configurations
├── runs/           # trained runs, checkpoints and TensorBoard logs
├── scripts/        # training, evaluation and utility entry points
├── src/            # models, classifiers, data loading and evaluation code
├── requirements.txt
└── README.md
```

The full training runs normally also contain logs and evaluation outputs. Training runs, checkpoints and TensorBoard logs are not included in this repository.
New training runs are stored under `runs/`.

## Configuration

Training and evaluation settings are defined in the YAML files inside `configs/`. Each configuration specifies the model, dataset sources, training parameters and inference settings.

Before training, split the labeled dataset into train, validation and test sets with:

```bash
python -m scripts.split_labeled \
  --labeled-dir <path_to_labeled_data> \
  --val-split 0.2 \
  --test-split 0.2 \
  --output-name split_mapping_60_20_20.csv
```

The generated split mapping is referenced in the corresponding YAML configuration. Additional datasets, for example data created with the Pool Annotation Tool, can be added as further data sources and assigned individually to train, validation or test.

## Training

Start a training run with one of the YAML configurations:

```bash
python -m scripts.train --config configs/<config>.yaml
```

The resulting run is stored under:

```text
runs/<run_name>/
```

## Evaluation

Evaluate an existing run with:

```bash
python -m scripts.evaluate \
  --run runs/<run_name> \
  --split test
```

Threshold and minimum mask area can optionally be overwritten:

```bash
python -m scripts.evaluate \
  --run runs/<run_name> \
  --split test \
  --threshold 0.3 \
  --min-area 30
```

The evaluation uses the checkpoint stored at `runs/<run_name>/weights/best.pt`.

Evaluation results are stored as versioned evaluations so that different inference settings can be compared without overwriting previous results.

## TensorBoard

Start TensorBoard from the `ml_pipeline` directory:

```bash
bash scripts/start_tensorboard.sh
```

Logs are read from `runs/` by default. If TensorBoard runs on a remote server, forward port `6006` and open `http://localhost:6006` locally.
