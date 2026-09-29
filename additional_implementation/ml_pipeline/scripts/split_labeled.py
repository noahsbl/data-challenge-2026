import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-d", "--labeled-dir", type=Path, default=Path("../data/labeled")
    )
    parser.add_argument(
        "-vs",
        "--val-split",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "-ts",
        "--test-split",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=0,
    )
    parser.add_argument(
        "-o",
        "--output-name",
        type=str,
        default="split_mapping.csv",
        help="Name der Ausgabedatei, z. B. split_mapping_80_10_10.csv",
    )

    args = parser.parse_args()

    labeled_dir = args.labeled_dir.expanduser()

    if args.val_split + args.test_split >= 1:
        raise ValueError("val_split + test_split must be smaller than 1.")

    if not args.output_name.endswith(".csv"):
        args.output_name += ".csv"

    csv_path = labeled_dir / "image_info.csv"

    df = pd.read_csv(csv_path)

    train_val, test = train_test_split(
        df,
        test_size=args.test_split,
        random_state=args.seed,
        stratify=df["contains_pool"],
    )

    relative_val_split = args.val_split / (1 - args.test_split)

    train, val = train_test_split(
        train_val,
        test_size=relative_val_split,
        random_state=args.seed,
        stratify=train_val["contains_pool"],
    )

    train = train.copy()
    val = val.copy()
    test = test.copy()

    train["split"] = "train"
    val["split"] = "val"
    test["split"] = "test"

    result = pd.concat([train, val, test]).sort_index()

    output_path = labeled_dir / args.output_name

    result.to_csv(
        output_path,
        index=False,
    )

    summary = pd.crosstab(
        result["split"],
        result["contains_pool"],
    )

    summary = summary.rename(
        columns={
            False: "false",
            True: "true",
        }
    )

    for column in ["false", "true"]:
        if column not in summary.columns:
            summary[column] = 0

    summary["total"] = summary["false"] + summary["true"]

    summary["prevalence"] = (summary["true"] / summary["total"] * 100).round(2).astype(
        str
    ) + "%"

    summary = summary[
        [
            "total",
            "true",
            "false",
            "prevalence",
        ]
    ]

    summary = summary.reset_index()
    summary.columns.name = None

    print(summary)
    print()
    print(f"Saved split mapping to: {output_path}")


if __name__ == "__main__":
    main()
