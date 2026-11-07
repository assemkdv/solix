"""
Example 4 -- preparing real-world inputs.

Real inputs rarely arrive in the package's schema. Two helpers convert them:

- build_index_from_hmi_filenames(): a headerless list of HMI file paths
  -> (timestamp, image_path) index.
  Non-image paths are excluded and reported, not silently dropped.
- adapt_goes_catalog(): a GOES catalog with start_time, peak_time,
  end_time, goes_class, noaa_active_region (an integrated GOES flare
  catalog) -> the catalog schema the dataset classes read.

Run:  python examples/04_loaders.py
"""

from pathlib import Path

import flare_indexer as fidx

DATA = Path(__file__).parent / "data"
OUTPUT = Path(__file__).parent / "output"


def main():
    OUTPUT.mkdir(exist_ok=True)

    index, report = fidx.build_index_from_hmi_filenames(
        DATA / "hmi_image_list.txt",
        output_path=OUTPUT / "04_index.csv",
        strict=False,              # exclude non-image rows instead of raising
        duplicate_policy="error",  # stop if two images share a timestamp
        return_report=True,
    )
    print("Image list -> index")
    print(f"  input rows: {report.total_input_rows}, valid: {report.valid_image_rows_before_dedup}, "
          f"excluded: {report.excluded_row_count} {report.excluded_rows}")
    print(index.to_string(index=False))

    catalog = fidx.adapt_goes_catalog(DATA / "goes_flares_sample.csv", OUTPUT / "04_flare_catalog.csv")
    print("\nGOES catalog -> flare catalog (extra columns such as ssw_flare_id are dropped)")
    print(catalog.to_string(index=False))

    labels = fidx.FullDiskSingleInstance(
        prediction_window=24, reducer=fidx.MaxFluxReducer(), labeler=fidx.BinaryThresholdLabeler("M"),
        include_metadata=True,
    ).build(OUTPUT / "04_index.csv", OUTPUT / "04_flare_catalog.csv")
    print("\nLabels for the converted inputs")
    print(labels.to_string(index=False))


if __name__ == "__main__":
    main()
