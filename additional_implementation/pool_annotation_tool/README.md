# Pool Annotation Tool

A small Streamlit tool for reviewing aerial images and predicted pool segmentation masks.

Each sample can be marked as:

- `accepted`
- `rejected`
- `edit`
- `unreviewed`

Reviews are stored in `review_status.csv` and can be changed later.

## Installation

From this directory:

```bash
pip install -r requirements.txt
```

## Data

Expected structure:

```text
model_labeled/
├── image_info.csv
├── imgs/
└── msks/
```

`image_info.csv` must contain at least:

```text
filename
mask_path
```

## Run

Set the dataset path in `run_annotation_tool.py`:

```python
DATA_DIR = Path("../data/model_labeled")
```

Then start the tool:

```bash
python run_annotation_tool.py
```

Streamlit uses port `8501` by default. For remote use, forward the port and open `http://localhost:8501` locally.

## Review workflow

The interface shows the RGB image, mask overlay and current review status. Use the buttons to accept, reject or mark a sample for manual editing.

Samples marked as `edit` are copied to `manual_edit/`. Review progress is saved automatically in `review_status.csv`.

## Export reviewed data

Accepted and rejected samples can be exported into a new dataset with:

```bash
python copy_reviewed_files.py \
  --data-dir /path/to/model_labeled \
  --output-dir /path/to/reviewed_dataset
```

Accepted samples are exported as positive examples. Rejected samples are exported as negative examples and receive an empty mask by default.
