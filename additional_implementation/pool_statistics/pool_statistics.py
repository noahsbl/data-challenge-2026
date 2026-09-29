import argparse
import os

import cv2
import numpy as np
import pandas as pd
import tifffile as tiff
from tqdm import tqdm


PIXEL_SIZE = 0.20
PIXEL_AREA = PIXEL_SIZE ** 2


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calculate pool statistics from segmentation masks."
    )
    parser.add_argument(
        "--data-dir",
        required=True,
        help="Path to the dataset directory containing msks/ and image_info.csv.",
    )
    return parser.parse_args()


def estimate_depth(area_m2):
    """Estimate the average water depth based on the water surface area."""

    if area_m2 < 3:
        return 0.35
    if area_m2 < 8:
        return 0.60
    if area_m2 < 20:
        return 0.90
    if area_m2 < 40:
        return 1.25
    if area_m2 < 60:
        return 1.45

    return None


def volume_group(volume_m3, area_m2):
    """Assign a pool to a volume group."""

    if area_m2 >= 60:
        return "Swimming Pool"

    if volume_m3 < 1:
        return "Paddling Pool"
    if volume_m3 < 8:
        return "Small"
    if volume_m3 < 25:
        return "Medium"
    if volume_m3 < 60:
        return "Large"

    return "Very Large"


def detect_shape(contour, area):
    """Classify the pool shape as circular, rectangular or other."""

    perimeter = cv2.arcLength(contour, True)

    if perimeter == 0:
        return "Other"

    circularity = 4 * np.pi * area / (perimeter * perimeter)

    _, _, width, height = cv2.boundingRect(contour)

    aspect_ratio = max(width, height) / max(1, min(width, height))
    rectangle_area = width * height
    extent = area / rectangle_area

    if circularity >= 0.80:
        return "Circular"

    if extent >= 0.75 and aspect_ratio <= 2.5:
        return "Rectangular"

    return "Other"


def main() -> None:
    arguments = parse_arguments()

    data_dir = arguments.data_dir
    mask_dir = os.path.join(data_dir, "msks")
    image_info_path = os.path.join(data_dir, "image_info.csv")
    output_csv = os.path.join(data_dir, "pool_statistics.csv")

    image_info = pd.read_csv(image_info_path)
    image_info = image_info.set_index("filename")

    mask_files = sorted(
        [
            filename
            for filename in os.listdir(mask_dir)
            if filename.lower().endswith((".tif", ".tiff"))
            and filename != "empty_mask.tif"
        ]
    )

    results = []

    summary_groups = {
        "Paddling Pool": 0,
        "Small": 0,
        "Medium": 0,
        "Large": 0,
        "Very Large": 0,
        "Swimming Pool": 0,
    }

    summary_shapes = {
        "Circular": 0,
        "Rectangular": 0,
        "Other": 0,
    }

    summary_regions = {}
    summary_years = {}

    total_volume = 0.0

    print(f"Analyzing {len(mask_files)} masks...")

    for filename in tqdm(mask_files):
        image_filename = filename.replace("MSK_", "", 1)

        info = image_info.loc[image_filename]

        region = info["bildflugname"]

        if region not in summary_regions:
            summary_regions[region] = {
                "Pools": 0,
                "Volume_m3": 0.0,
            }

        year = str(pd.to_datetime(info["aktualitaet"]).year)

        if year not in summary_years:
            summary_years[year] = 0

        mask = tiff.imread(
            os.path.join(mask_dir, filename)
        )

        mask = (mask > 0).astype(np.uint8) * 255

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask,
            connectivity=8,
        )

        for label in range(1, num_labels):
            component = np.zeros_like(mask)
            component[labels == label] = 255

            contours, _ = cv2.findContours(
                component,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )

            if len(contours) == 0:
                continue

            contour = contours[0]

            area_pixels = stats[label, cv2.CC_STAT_AREA]
            area_m2 = area_pixels * PIXEL_AREA

            shape = detect_shape(
                contour,
                area_pixels,
            )

            summary_shapes[shape] += 1

            depth = estimate_depth(area_m2)

            if depth is None:
                volume = np.nan
            else:
                volume = area_m2 * depth
                total_volume += volume

            group = volume_group(
                0 if np.isnan(volume) else volume,
                area_m2,
            )

            summary_groups[group] += 1
            summary_regions[region]["Pools"] += 1
            summary_years[year] += 1

            results.append(
                {
                    "Image": image_filename,
                    "Region": info["bildflugname"],
                    "Date": info["aktualitaet"],
                    "Area (m²)": round(area_m2, 2),
                    "Estimated Depth (m)": (
                        None if depth is None else round(depth, 2)
                    ),
                    "Estimated Volume (m³)": (
                        None if np.isnan(volume) else round(volume, 2)
                    ),
                    "Estimated Volume (L)": (
                        None if np.isnan(volume) else int(round(volume * 1000))
                    ),
                    "Shape": shape,
                    "Group": group,
                }
            )

    result_df = pd.DataFrame(results)

    if not result_df.empty:
        result_df = result_df.sort_values(
            by=[
                "Region",
                "Image",
                "Estimated Volume (m³)",
            ],
            ascending=[
                True,
                True,
                False,
            ],
        )

    result_df.to_csv(
        output_csv,
        index=False,
    )

    print("\n========================================")
    print("Pool Statistics")
    print("========================================")

    print(f"Masks: {len(mask_files)}")
    print(f"Pools: {len(result_df)}")

    print("\n----------- Shapes -----------")
    for key, value in summary_shapes.items():
        print(f"{key:15}: {value}")

    print("\n----------- Years -----------")
    for year, count in sorted(summary_years.items()):
        print(f"{year:15}: {count} pools")

    print("\n---------- Regions ----------")
    for region, values in sorted(summary_regions.items()):
        print(f"{region:20}: {values['Pools']} pools")

    print("\n------ Volume Groups ------")
    for key, value in summary_groups.items():
        print(f"{key:15}: {value}")

    print("\n---------- Overall ----------")

    calculated_pools = result_df[
        result_df["Estimated Volume (m³)"].notna()
    ]

    if len(calculated_pools) > 0:
        print(f"Calculated pools : {len(calculated_pools)}")
        print(
            f"Average area     : "
            f"{calculated_pools['Area (m²)'].mean():.2f} m²"
        )
        print(
            f"Average depth    : "
            f"{calculated_pools['Estimated Depth (m)'].mean():.2f} m"
        )
        print(
            f"Average volume   : "
            f"{calculated_pools['Estimated Volume (m³)'].mean():.2f} m³"
        )

    print("\nEstimated total volume:")
    print(f"{total_volume:.2f} m³")
    print(f"{total_volume * 1000:,.0f} liters")

    print("\nCSV saved to:")
    print(output_csv)


if __name__ == "__main__":
    main()