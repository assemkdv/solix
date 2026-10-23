"""
Acquire the real images and flare catalog used by examples 07, 08, 10 and 11.

This script only acquires data; it does no labeling. It writes:

- images/*.jp2                 SDO/HMI line-of-sight magnetograms from the
                               Helioviewer archive (cached, not in git).
- image_manifest.csv           one row per requested time: the Helioviewer
                               image id and URL, the observation time read
                               from the JP2's own FITS header, and the
                               file's SHA-256.
- hek_swpc_flares.csv          the raw HEK query result (SWPC GOES flare list).
- flare_catalog.csv            the same flares in flare_indexer's catalog schema.
- catalog_provenance.json      the HEK query and the interval it covered.

Three modes:

- First acquisition (no image_manifest.csv yet): discover the image
  closest to each requested time with Helioviewer's getClosestImage,
  download it, and write the manifest.
- Replay (default when image_manifest.csv exists): use only the image ids,
  URLs and SHA-256 recorded in the manifest. getClosestImage is not called,
  rows recorded as missing or duplicate are not downloaded, and every
  cached or downloaded file must match its recorded SHA-256. On a mismatch
  the script stops; neither the file nor the recorded hash is replaced.
  Only fields read from the verified files' headers are rewritten.
- Refresh (--refresh): run discovery again and rewrite the manifest. The
  archive may return different images than before.

Image timestamps are DATE_OBS from inside each downloaded file at full
precision, never the requested time and never rounded. A request whose
closest archived image is more than MAX_OFFSET away is recorded as
missing, and a request that returns an image already used for an earlier
request is recorded as a duplicate; neither is used in place of anything.

Run:  python examples/real_data/fetch_real_data.py              # replay (or first acquisition)
      python examples/real_data/fetch_real_data.py --refresh    # discover images again
Needs network access only for files that are not cached yet; uses only the
standard library and pandas.
"""

import argparse
import hashlib
import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

from hmi_jp2 import matches_api_date, observation_time, read_jp2_header

HERE = Path(__file__).resolve().parent
MANIFEST_NAME = "image_manifest.csv"
RAW_CATALOG = HERE / "hek_swpc_flares.csv"
CATALOG = HERE / "flare_catalog.csv"
CATALOG_PROVENANCE = HERE / "catalog_provenance.json"

HELIOVIEWER = "https://api.helioviewer.org/v2"
HMI_MAGNETOGRAM_SOURCE_ID = 19  # Helioviewer source id for "SDO HMI magnetogram"
HEK = "https://www.lmsal.com/hek/her"

# Ten hourly requests. The first M-class flare of this period starts on
# 2017-09-04 at 05:36, so 24-hour windows from these images give both labels.
REQUESTED_TIMES = pd.date_range("2017-09-03 01:00:00", periods=10, freq="1h")
MAX_OFFSET = pd.Timedelta("60s")

# The catalog interval must contain every prediction window (t, t + 24h)
# of every image above; examples 07, 08, 10 and 11 check this before labeling.
CATALOG_START = pd.Timestamp("2017-09-01 00:00:00")
CATALOG_END = pd.Timestamp("2017-09-06 00:00:00")

USER_AGENT = "flare-indexer-examples/0.2.1"

# Columns read from the image's own header; replay may rewrite these.
HEADER_COLUMNS = (
    "timestamp", "date_obs", "offset_seconds", "t_obs", "t_rec", "telescop",
    "instrume", "content", "wavelnth", "quality", "crota2", "bytes",
)


class HashMismatchError(RuntimeError):
    """A file does not match the SHA-256 recorded in the manifest."""


def _get(url: str, timeout: int = 120) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _format_time(timestamp: pd.Timestamp) -> str:
    return timestamp.isoformat(sep=" ", timespec="microseconds")


def _download_verified(url: str, path: Path, expected_sha256: str) -> None:
    """Download url to path only if the content matches expected_sha256."""
    data = _get(url)
    actual = _sha256(data)
    if actual != expected_sha256:
        raise HashMismatchError(
            f"Downloaded {url} has SHA-256 {actual}, but the manifest records {expected_sha256}. "
            f"Nothing was written to {path} and the manifest was not changed. The archive may "
            "have changed; run with --refresh to discover images again."
        )
    partial = path.with_suffix(path.suffix + ".part")
    partial.write_bytes(data)
    partial.replace(path)


def _header_fields(path: Path, requested: pd.Timestamp) -> dict:
    header = read_jp2_header(path)
    observed = observation_time(header)
    fields = {
        "timestamp": _format_time(observed),
        "date_obs": header["DATE_OBS"],
        "offset_seconds": (observed - requested).total_seconds(),
        "t_obs": header["T_OBS"],
        "t_rec": header["T_REC"],
        "telescop": header["TELESCOP"],
        "instrume": header["INSTRUME"],
        "content": header["CONTENT"],
        "wavelnth": header["WAVELNTH"],
        "quality": header["QUALITY"],
        "crota2": header["CROTA2"],
        "bytes": path.stat().st_size,
    }
    return {key: "" if value is None else value for key, value in fields.items()}, header


def _print_row(row: dict, action: str) -> None:
    print(f"{row['requested_time']}  {action:10s}  id={row['helioviewer_id']}  DATE_OBS={row['date_obs']}  "
          f"offset={float(row['offset_seconds']):+.1f}s  {row['status']}")


def discover(root: Path = HERE) -> pd.DataFrame:
    """
    First acquisition or explicit refresh: ask getClosestImage for each
    requested time, download, read the header, and write a new manifest.
    A file already cached under an id the old manifest records must still
    match the recorded SHA-256.
    """
    images, manifest_path = root / "images", root / MANIFEST_NAME
    images.mkdir(exist_ok=True)
    previous = pd.read_csv(manifest_path, dtype=str, keep_default_na=False) if manifest_path.exists() else None
    known_sha = {} if previous is None else dict(zip(previous["helioviewer_id"], previous["sha256"]))

    rows = []
    usable_ids, usable_times = {}, {}
    for requested in REQUESTED_TIMES:
        query = urllib.parse.urlencode({
            "date": requested.strftime("%Y-%m-%dT%H:%M:%SZ"), "sourceId": HMI_MAGNETOGRAM_SOURCE_ID,
        })
        closest = json.loads(_get(f"{HELIOVIEWER}/getClosestImage/?{query}"))
        image_id = str(closest["id"])
        url = f"{HELIOVIEWER}/getJP2Image/?id={image_id}"
        path = images / f"hmi_magnetogram_{image_id}.jp2"
        row = {
            "requested_time": _format_time(requested),
            "status": "",
            "duplicate_of": "",
            "image_path": "",
            "helioviewer_id": image_id,
            "helioviewer_date": closest["date"],
            "width": closest["width"],
            "height": closest["height"],
            "source_url": url,
        }

        if image_id in usable_ids:
            # The same archived image answered two requests: keep it for the
            # first one only. Beyond MAX_OFFSET it is missing for this
            # request, as for any other image; otherwise a duplicate.
            first = usable_ids[image_id]
            row.update({key: first[key] for key in HEADER_COLUMNS + ("sha256",)})
            offset = pd.Timestamp(first["timestamp"]) - requested
            row["offset_seconds"] = offset.total_seconds()
            if abs(offset) > MAX_OFFSET:
                row["status"] = "missing"
            else:
                row.update(status="duplicate", duplicate_of=first["requested_time"])
            _print_row(row, "reused id")
            rows.append(row)
            continue

        if path.exists() and image_id in known_sha:
            if _sha256(path.read_bytes()) != known_sha[image_id]:
                raise HashMismatchError(
                    f"Cached {path} does not match the SHA-256 recorded for id {image_id}; "
                    "the file and manifest were left unchanged. Delete the file to download it again."
                )
            action = "cached"
        elif image_id in known_sha:
            _download_verified(url, path, known_sha[image_id])
            action = "downloaded"
        else:
            path.write_bytes(_get(url))
            action = "downloaded"

        fields, header = _header_fields(path, requested)
        if not matches_api_date(header, closest["date"]):
            raise ValueError(f"image {image_id}: header DATE_OBS {header['DATE_OBS']} does not round "
                             f"to Helioviewer date {closest['date']}")
        row.update(fields, sha256=_sha256(path.read_bytes()))
        observed = pd.Timestamp(row["timestamp"])

        if abs(observed - requested) > MAX_OFFSET:
            row["status"] = "missing"
            path.unlink()
        elif observed in usable_times:
            row.update(status="duplicate", duplicate_of=usable_times[observed]["requested_time"])
        else:
            row.update(status="ok", image_path=f"images/{path.name}")
            usable_ids[image_id] = usable_times[observed] = row
        _print_row(row, action)
        rows.append(row)

    manifest = pd.DataFrame(rows)
    manifest.to_csv(manifest_path, index=False)
    return manifest


def replay(root: Path = HERE) -> pd.DataFrame:
    """
    Re-create the cached images from the manifest alone: recorded ids,
    URLs and SHA-256 only, no getClosestImage, nothing downloaded for rows
    that are not "ok". Rewrites only the HEADER_COLUMNS of verified files.
    """
    manifest_path = root / MANIFEST_NAME
    manifest = pd.read_csv(manifest_path, dtype=str, keep_default_na=False)
    if "duplicate_of" not in manifest.columns:
        manifest.insert(manifest.columns.get_loc("status") + 1, "duplicate_of", "")
    (root / "images").mkdir(exist_ok=True)

    for i, row in manifest.iterrows():
        if row["status"] != "ok":
            _print_row(row, "skipped")
            continue
        path = root / row["image_path"]
        if path.exists():
            actual = _sha256(path.read_bytes())
            if actual != row["sha256"]:
                raise HashMismatchError(
                    f"Cached {path} has SHA-256 {actual}, but the manifest records {row['sha256']}. "
                    "The file and manifest were left unchanged. Delete the file to download it again."
                )
            action = "cached"
        else:
            _download_verified(row["source_url"], path, row["sha256"])
            action = "downloaded"

        fields, header = _header_fields(path, pd.Timestamp(row["requested_time"]))
        if not matches_api_date(header, row["helioviewer_date"]):
            raise ValueError(f"{path}: header DATE_OBS {header['DATE_OBS']} does not round to the "
                             f"recorded Helioviewer date {row['helioviewer_date']}")
        if abs(pd.Timedelta(seconds=fields["offset_seconds"])) > MAX_OFFSET:
            raise ValueError(f"{path}: observed {fields['timestamp']} is more than {MAX_OFFSET} from the request")
        for key, value in fields.items():
            if str(value) != str(row[key]):
                print(f"    {key}: {row[key]} -> {value} (read from the file header)")
            manifest.at[i, key] = str(value)
        _print_row(manifest.loc[i], action)

    manifest.to_csv(manifest_path, index=False)
    return manifest


def fetch_catalog() -> pd.DataFrame:
    params = {
        "cosec": 2, "cmd": "search", "type": "column", "event_type": "FL",
        "event_starttime": CATALOG_START.strftime("%Y-%m-%dT%H:%M:%S"),
        "event_endtime": CATALOG_END.strftime("%Y-%m-%dT%H:%M:%S"),
        "event_coordsys": "helioprojective", "x1": -5000, "x2": 5000, "y1": -5000, "y2": 5000,
        "param0": "FRM_NAME", "op0": "=", "value0": "SWPC",
        "return": "event_starttime,event_peaktime,event_endtime,fl_goescls,ar_noaanum,frm_name,kb_archivid",
        "result_limit": 1000,
    }
    url = f"{HEK}?{urllib.parse.urlencode(params)}"
    response = json.loads(_get(url, timeout=300))
    if response.get("overmax"):
        raise RuntimeError("HEK returned more results than result_limit; the catalog would be incomplete")

    raw = pd.DataFrame(response["result"]).sort_values(["event_starttime", "event_peaktime"])
    raw = raw.drop_duplicates("kb_archivid")
    raw.to_csv(RAW_CATALOG, index=False)

    no_class = raw["fl_goescls"].fillna("").str.strip() == ""
    catalog = pd.DataFrame({
        "start_time": pd.to_datetime(raw["event_starttime"]),
        "peak_time": pd.to_datetime(raw["event_peaktime"]),
        "end_time": pd.to_datetime(raw["event_endtime"]),
        "class": raw["fl_goescls"].str.strip(),
        "active_region": raw["ar_noaanum"].where(raw["ar_noaanum"] > 0).astype("Int64"),
    })[~no_class.to_numpy()]
    catalog.to_csv(CATALOG, index=False)

    CATALOG_PROVENANCE.write_text(json.dumps({
        "source": "Heliophysics Event Knowledgebase (HEK), FRM_NAME=SWPC (NOAA SWPC GOES X-ray flare list)",
        "query_url": url,
        "query_start": str(CATALOG_START),
        "query_end": str(CATALOG_END),
        "raw_rows": int(len(raw)),
        "rows_without_goes_class_dropped": int(no_class.sum()),
        "catalog_rows": int(len(catalog)),
        "note": "HEK returns events overlapping [query_start, query_end]. The list records flares; "
                "it does not certify that GOES observed without gaps (docs/decisions.md, Q5).",
    }, indent=2) + "\n")
    print(f"HEK SWPC flares {CATALOG_START} .. {CATALOG_END}: {len(raw)} rows, "
          f"{int(no_class.sum())} without a GOES class dropped, {len(catalog)} written to {CATALOG.name}")
    return catalog


def main(argv=None, root: Path = HERE):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--refresh", action="store_true",
                        help="discover the closest images again with getClosestImage and rewrite the manifest")
    parser.add_argument("--refresh-catalog", action="store_true", help="query HEK even if flare_catalog.csv exists")
    args = parser.parse_args(argv)

    if args.refresh or not (root / MANIFEST_NAME).exists():
        print("Discovering the closest images (getClosestImage)")
        manifest = discover(root)
    else:
        print(f"Replaying {MANIFEST_NAME} (recorded ids, URLs and SHA-256; pass --refresh to discover again)")
        manifest = replay(root)
    print(f"{(manifest['status'] == 'ok').sum()} of {len(manifest)} requested images available")

    if args.refresh_catalog or not CATALOG.exists():
        fetch_catalog()
    else:
        print(f"{CATALOG.name} exists; pass --refresh-catalog to query HEK again")


if __name__ == "__main__":
    sys.exit(main())
