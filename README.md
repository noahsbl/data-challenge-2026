# Data Challenge 2026

This project was developed as part of the Data Challenge 2026 in cooperation with the OOWV. The task was to automatically detect and segment private swimming pools in aerial imagery.

Several approaches were implemented and evaluated, including ResNet, U-Net and SegFormer. SegFormer was used as the final submitted model.

In addition to the final inference pipeline, this repository contains the training and evaluation pipeline, tooling for model-assisted labeling and manual annotation review, as well as an additional implementation for calculating pool statistics.

## Structure

```text
.
├── submitted_model_inference_pipeline/
│   └── Final SegFormer inference and evaluation pipeline
└── additional_implementation/
    ├── ml_pipeline/
    │   └── Training and evaluation pipeline for ResNet, U-Net and SegFormer
    ├── model_labeling/
    │   └── Model-assisted labeling of unlabeled images
    ├── pool_annotation_tool/
    │   └── Review of model-labeled data
    └── pool_statistics/
        └── Statistical analysis of pool segmentation masks
```

More detailed documentation for the individual components can be found in their respective READMEs:

- [Submitted Model Inference Pipeline](submitted_model_inference_pipeline/README.md)
- [ML Pipeline](additional_implementation/ml_pipeline/README.md)
- [Model Labeling](additional_implementation/model_labeling/README.md)
- [Pool Annotation Tool](additional_implementation/pool_annotation_tool/README.md)
- [Pool Statistics](additional_implementation/pool_statistics/README.md)

## Installation

Clone the repository and install the required dependencies:

```bash
pip install -r requirements.txt
```

Additional component-specific dependencies and setup instructions are described in the respective READMEs.

## Components

### Submitted Model Inference Pipeline

The [submitted model inference pipeline](submitted_model_inference_pipeline/README.md) contains the final SegFormer model and the inference and evaluation pipeline used for the challenge submission.

### ML Pipeline

The [ML pipeline](additional_implementation/ml_pipeline/README.md) contains the implementations for training and evaluating the different model approaches, including ResNet, U-Net and SegFormer.

### Model Labeling

The [model labeling pipeline](additional_implementation/model_labeling/README.md) applies a trained segmentation model to unlabeled data and exports predicted positive samples and their segmentation masks for further review.

### Pool Annotation Tool

The [pool annotation tool](additional_implementation/pool_annotation_tool/README.md) provides an interface for manually reviewing and correcting model-generated annotations.

### Pool Statistics

The [pool statistics implementation](additional_implementation/pool_statistics/README.md) performs additional statistical analyses based on the generated pool segmentation masks.

## Dataset

The aerial imagery and annotations provided for the Data Challenge are **not included in this repository**.

The code expects the corresponding data to be provided locally. Refer to the component-specific READMEs for the expected directory structures and configuration.

## Authors

Developed by Data Challenge 2026 Group 11:
- Noah Saibel
- Marcel Weber

## License

This project is licensed under the [MIT License](LICENSE).