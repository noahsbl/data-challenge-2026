# Pool Segmentation: SegFormer Inference Pipeline

This repository contains the submitted SegFormer model and the given inference/evaluation pipeline for pool segmentation.

## Installation

Install the required Python packages with:

```bash
pip install -r requirements.txt
```

## Data

Set `DATA_DIR` in `pipeline.py` to the directory containing the labeled dataset. The expected structure is:

```text
DATA_DIR/
├── imgs/
├── msks/
└── image_info.csv
```

## Model configuration

The submitted model is initialized in `pipeline.py` as follows:

```python
from classifiers.segformer_segmentation_classifier import SegFormerSegmentationClassifier

classifier = [
    SegFormerSegmentationClassifier(
        model_path=os.path.join("classifiers", "segformer_segmentation_classifier", "segformer_iou_optimized.pt"),
        threshold=0.3,
        image_size=512,
        device="cpu",
        min_area=30,
    )
]
```

## Run

Navigate to the `submitted_model_inference_pipeline` directory:

```bash
cd submitted_model_inference_pipeline
```

Execute the evaluation pipeline from the project root:

```bash
python pipeline.py
```

The pipeline loads the images and reference masks, runs the SegFormer classifier and reports the configured binary and segmentation metrics.