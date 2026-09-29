# Model Labeling

This tool applies a trained SegFormer model to unlabeled images and exports positive predictions together with their generated segmentation masks.

## Usage

Run from the `pseudo_labeling` directory:

```bash
python classify_unlabeled.py \
  --model-path ../ml_pipeline/runs/<run_name>/weights/best.pt \
  --threshold 0.3 \
  --min-area 30 \
  --device cpu
```