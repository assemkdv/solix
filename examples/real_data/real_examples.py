"""
Helpers shared by the real-image examples (06, 07, 10 with PyTorch; 08,
09, 11 with TensorFlow): turning the download manifests into image indexes,
checking catalog coverage, saving sequence tables with list-valued columns,
and an independent label check that does not use flare_indexer.
"""

import hashlib
import json
import sys
from decimal import Decimal
from pathlib import Path

import pandas as pd

REAL_DATA = Path(__file__).resolve().parent
MANIFEST = REAL_DATA / "image_manifest.csv"
CATALOG = REAL_DATA / "flare_catalog.csv"
CATALOG_PROVENANCE = REAL_DATA / "catalog_provenance.json"
FETCH_COMMAND = "python examples/real_data/fetch_real_data.py"
SHARP_MANIFEST = REAL_DATA / "sharp_manifest.csv"
SHARP_FETCH_COMMAND = "python examples/real_data/fetch_sharp_patches.py --email you@example.com"


def require_optional_dependencies(images: str = "jp2"):
    """
    Exit with a clear message if the image reader is missing: Pillow with
    JPEG 2000 for the full-disk images (images="jp2"), astropy and Pillow
    for the SHARP patches (images="fits"). The framework itself (torch or
    tensorflow) is imported at the top of each example.
    """
    missing = []
    try:
        from PIL import features
        if images == "jp2" and not features.check("jpg_2000"):
            missing.append("Pillow built with OpenJPEG (JPEG 2000 support)")
    except ImportError:
        missing.append("pillow")
    if images == "fits":
        try:
            import astropy  # noqa: F401
        except ImportError:
            missing.append("astropy")
    if missing:
        extra = " astropy" if images == "fits" else ""
        sys.exit(f"This example needs optional dependencies: {', '.join(missing)}. "
                 f"Install them with:  pip install pillow{extra}")


def build_image_index(output_path: Path) -> pd.DataFrame:
    """
    Write the image index flare_indexer reads: one row per available image,
    timestamp = full-precision DATE_OBS from the image's own header
    (recorded in the manifest, never rounded), image_path = path relative to examples/real_data.
    Requests recorded as missing are left out. Exits if a file is absent or
    no longer matches the SHA-256 recorded when it was downloaded.
    """
    if not MANIFEST.exists():
        sys.exit(f"{MANIFEST} not found. Download the data first:  {FETCH_COMMAND}")
    manifest = pd.read_csv(MANIFEST)
    available = manifest[manifest["status"] == "ok"]

    for path, sha256 in zip(available["image_path"], available["sha256"]):
        file = REAL_DATA / path
        if not file.exists():
            sys.exit(f"{file} not found. Download the images first:  {FETCH_COMMAND}")
        if hashlib.sha256(file.read_bytes()).hexdigest() != sha256:
            sys.exit(f"{file} does not match the SHA-256 in {MANIFEST.name}; run {FETCH_COMMAND}")

    timestamps = pd.to_datetime(available["timestamp"])
    header_times = pd.to_datetime(available["date_obs"], utc=True).dt.tz_localize(None)
    if not (timestamps == header_times).all():
        sys.exit(f"{MANIFEST.name}: timestamps differ from the full-precision DATE_OBS; run {FETCH_COMMAND}")

    index = pd.DataFrame({
        "timestamp": timestamps,
        "image_path": available["image_path"],
    }).sort_values("timestamp")
    index.to_csv(output_path, index=False)
    return index


def build_patch_index(output_path: Path) -> pd.DataFrame:
    """
    Write the active-region image index: one row per SHARP patch, with
    timestamp = DATE-OBS (UTC) from the patch's header, active_region = the
    NOAA number from the header, image_path relative to examples/real_data.
    Exits if a patch is absent or no longer matches its recorded SHA-256.
    """
    if not SHARP_MANIFEST.exists():
        sys.exit(f"{SHARP_MANIFEST} not found. Download the patches first:  {SHARP_FETCH_COMMAND}")
    manifest = pd.read_csv(SHARP_MANIFEST)
    for path, sha256 in zip(manifest["image_path"], manifest["sha256"]):
        file = REAL_DATA / path
        if not file.exists():
            sys.exit(f"{file} not found. Download the patches first:  {SHARP_FETCH_COMMAND}")
        if hashlib.sha256(file.read_bytes()).hexdigest() != sha256:
            sys.exit(f"{file} does not match the SHA-256 in {SHARP_MANIFEST.name}; run {SHARP_FETCH_COMMAND}")

    index = pd.DataFrame({
        "timestamp": pd.to_datetime(manifest["timestamp"]),
        "active_region": manifest["noaa_ar"],
        "image_path": manifest["image_path"],
    }).sort_values(["timestamp", "active_region"])
    index.to_csv(output_path, index=False)
    return index


def check_catalog_coverage(reference_times, prediction_window) -> dict:
    """
    Stop unless every prediction window [t, t + prediction_window) lies
    inside the interval the catalog was queried for. This is the only
    coverage the example can establish: a flare list cannot show that the
    instrument observed without gaps.
    """
    provenance = json.loads(CATALOG_PROVENANCE.read_text())
    query_start, query_end = pd.Timestamp(provenance["query_start"]), pd.Timestamp(provenance["query_end"])
    window = pd.Timedelta(prediction_window)
    first, last = min(reference_times), max(reference_times) + window
    if first < query_start or last > query_end:
        raise SystemExit(
            f"Catalog queried for {query_start} .. {query_end} does not cover the prediction "
            f"windows {first} .. {last}. Widen CATALOG_START / CATALOG_END in fetch_real_data.py, then run "
            f"{FETCH_COMMAND} --refresh-catalog."
        )
    return provenance


def flares_in_window(catalog: pd.DataFrame, t: pd.Timestamp, prediction_window) -> pd.DataFrame:
    """
    Independent of flare_indexer: catalog rows whose start_time is in
    [t, t + prediction_window), the package's window rule.
    """
    end = t + pd.Timedelta(prediction_window)
    return catalog[(catalog["start_time"] >= t) & (catalog["start_time"] < end)]


_EXPONENT = {"A": -8, "B": -7, "C": -6, "M": -5, "X": -4}


def _at_least_m1(goes_class: str) -> bool:
    """Exact decimal check, written separately from flare_indexer: "C10.0" counts as M1.0."""
    text = goes_class.strip().upper()
    multiplier = Decimal(text[1:] or "1")
    return multiplier.scaleb(_EXPONENT[text[0]]) >= Decimal("1e-5")


def independent_m_label(catalog: pd.DataFrame, t: pd.Timestamp, prediction_window,
                        region=None) -> tuple[int, pd.DataFrame]:
    """
    1 if any flare of M1.0 or stronger starts in [t, t + prediction_window),
    else 0. With region, only that NOAA region's flares count.
    """
    flares = flares_in_window(catalog, t, prediction_window)
    if region is not None:
        # As numbers: a column with a blank region is read as floats (12673.0).
        flares = flares[pd.to_numeric(flares["active_region"]).astype("Int64") == int(region)]
    strong = flares[flares["class"].map(_at_least_m1)]
    return int(not strong.empty), strong


def date_obs_times() -> dict:
    """
    image_path -> the raw DATE_OBS header string, parsed as UTC at full
    precision. Used to check that index timestamps equal the header times.
    """
    manifest = pd.read_csv(MANIFEST)
    available = manifest[manifest["status"] == "ok"]
    times = pd.to_datetime(available["date_obs"], utc=True).dt.tz_localize(None)
    return dict(zip(available["image_path"], times))


def format_time(timestamp) -> str:
    """One lossless text form for timestamps in printed output and comparisons."""
    return pd.Timestamp(timestamp).isoformat(sep=" ", timespec="microseconds")


def save_sequences(sequences: pd.DataFrame, path: Path) -> None:
    """Write list-valued columns as JSON lists (timestamps as ISO strings)."""
    out = sequences.copy()
    out["timestamps"] = out["timestamps"].map(lambda ts: json.dumps([t.isoformat(sep=" ") for t in ts]))
    out["image_path"] = out["image_path"].map(json.dumps)
    out.to_csv(path, index=False)


def load_sequences(path: Path) -> pd.DataFrame:
    """Inverse of save_sequences."""
    sequences = pd.read_csv(path, parse_dates=["sequence_start", "sequence_end"])
    sequences["timestamps"] = sequences["timestamps"].map(lambda s: [pd.Timestamp(t) for t in json.loads(s)])
    sequences["image_path"] = sequences["image_path"].map(json.loads)
    return sequences


def check_sequence(row, cadence) -> None:
    """Raise if a sequence row's images are not `cadence` apart or do not end at sequence_end."""
    timestamps, paths, cadence = row["timestamps"], row["image_path"], pd.Timedelta(cadence)
    if len(timestamps) != len(paths) or len(paths) != row["n_images"]:
        raise ValueError(f"sequence ending {row['sequence_end']}: {len(timestamps)} timestamps, {len(paths)} paths")
    if any(later - earlier != cadence for earlier, later in zip(timestamps, timestamps[1:])):
        raise ValueError(f"sequence ending {row['sequence_end']}: images are not {cadence} apart: {timestamps}")
    if timestamps[-1] != row["sequence_end"]:
        raise ValueError(f"sequence ending {row['sequence_end']}: last image is at {timestamps[-1]}")


def load_catalog() -> pd.DataFrame:
    return pd.read_csv(CATALOG, parse_dates=["start_time", "peak_time", "end_time"])
