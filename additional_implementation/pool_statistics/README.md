# Pool Statistics

This tool calculates additional statistics from pool segmentation masks, including pool area, estimated volume, shape, region and acquisition year.

It can be used with predicted masks as well as model-labeled or manually corrected data.

## Usage

Run from the `pool_statistics` directory:

```bash
python pool_statistics.py \
  --data-dir <path_to_dataset>
```

The dataset directory must contain:

```text
<dataset>/
├── msks/
└── image_info.csv
```

The calculated statistics are saved to:

```text
<dataset>/pool_statistics.csv
```

The script assumes a ground resolution of 20 cm per pixel.
