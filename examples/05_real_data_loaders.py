"""
Example 5 -- preparing real-world inputs.

Real inputs rarely arrive in the package's schema. Two helpers convert them:

- build_image_index_from_filenames(): a headerless list of HMI image paths
  (like totalfiles_jpg_512.csv) -> (timestamp, image_path) image index.
  Non-image paths are excluded and reported, not silently dropped.
- adapt_goes_catalog(): a GOES catalog with start_time, peak_time,
  end_time, goes_class, noaa_active_region (like goes_flares_integrated.csv)
  -> the catalog schema DatasetBuilder reads.

Run:  python examples/05_real_data_loaders.py
"""

from pathlib import Path

import flare_indexer as fidx

DATA = Path(__file__).parent / "data"
OUTPUT = Path(__file__).parent / "output"


def main():
    OUTPUT.mkdir(exist_ok=True)

    index, report = fidx.build_image_index_from_filenames(
        DATA / "hmi_image_list.txt",
        output_path=OUTPUT / "05_image_index.csv",
        strict=False,              # exclude non-image rows instead of raising
        duplicate_policy="error",  # stop if two images share a timestamp
        return_report=True,
    )
    print("Image list -> image index")
    print(f"  input rows: {report.total_input_rows}, valid: {report.valid_image_rows_before_dedup}, "
          f"excluded: {report.excluded_row_count} {report.excluded_rows}")
    print(index.to_string(index=False))

    catalog = fidx.adapt_goes_catalog(DATA / "goes_flares_sample.csv", OUTPUT / "05_flare_catalog.csv")
    print("\nGOES catalog -> flare catalog (extra columns such as ssw_flare_id are dropped)")
    print(catalog.to_string(index=False))

    labels = fidx.DatasetBuilder(
        prediction_window=24, strategy=fidx.BinaryThresholdStrategy("M"),
        event_time="start", include_metadata=True,
    ).build(OUTPUT / "05_image_index.csv", OUTPUT / "05_flare_catalog.csv")
    print("\nLabels for the converted inputs")
    print(labels.to_string(index=False))


if __name__ == "__main__":
    main()
