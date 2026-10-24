"""
Download the active-region patches used by examples 10 and 11.

The patches are SDO/HMI SHARP magnetograms (hmi.sharp_cea_720s, segment
"magnetogram") of two regions, hourly from 2017-09-03 00:00 to 09:00 TAI:

    HARP 7115 = NOAA 12673 (M- and X-class flares on 2017-09-04 to 06)
    HARP 7120 = NOAA 12675 (no flares in these days)

The files are already in the repository (sharp_patches/), so the examples
run without this script. It records where they came from and lets anyone
download them again. It writes:

- sharp_patches/*.fits     one magnetogram per patch
- sharp_manifest.csv       one row per patch: HARP and NOAA numbers from
                           the file's own header, the record time (T_REC,
                           TAI), the observation time (DATE-OBS, UTC; used
                           as the image timestamp), the image size and the
                           file's SHA-256

JSOC only exports data to a registered email address (free:
http://jsoc.stanford.edu/ajax/register_email.html). Give it with --email
or the JSOC_EMAIL environment variable; it is not written anywhere.

Run:  python examples/real_data/fetch_sharp_patches.py --email you@example.com
Needs network access and: pip install drms astropy pandas
"""

import argparse
import hashlib
import os
import ssl
import sys
import urllib.request
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
PATCHES = HERE / "sharp_patches"
MANIFEST = HERE / "sharp_manifest.csv"

HARPS = (7115, 7120)
QUERY = "hmi.sharp_cea_720s[{harps}][2017.09.03_00:00:00_TAI-2017.09.03_09:00:00_TAI@1h]{{magnetogram}}"


def download(url: str, target: Path) -> None:
    """Fetch one exported file (JSOC redirects to https://jsoc1.stanford.edu)."""
    try:
        import certifi
        context = ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        context = ssl.create_default_context()
    with urllib.request.urlopen(url, timeout=120, context=context) as response:
        target.write_bytes(response.read())


def read_header(path: Path) -> tuple[dict, tuple]:
    from astropy.io import fits

    with fits.open(path) as hdus:
        hdu = next(h for h in hdus if h.data is not None)
        return dict(hdu.header), hdu.data.shape


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Download the SHARP patches for examples 10 and 11.")
    parser.add_argument("--email", default=os.environ.get("JSOC_EMAIL"),
                        help="email address registered at JSOC (or set JSOC_EMAIL)")
    args = parser.parse_args(argv)
    if not args.email:
        sys.exit("JSOC needs a registered email: pass --email or set JSOC_EMAIL "
                 "(register at http://jsoc.stanford.edu/ajax/register_email.html).")
    import drms

    query = QUERY.format(harps=",".join(map(str, HARPS)))
    print(f"Exporting {query} ...")
    export = drms.Client().export(query, method="url", protocol="fits", email=args.email)
    export.wait()
    if export.has_failed():
        sys.exit(f"JSOC export failed: {export.status}")

    PATCHES.mkdir(exist_ok=True)
    rows = []
    for url in export.urls["url"]:
        target = PATCHES / url.rsplit("/", 1)[1]
        download(url, target)
        header, shape = read_header(target)
        harp = int(header["HARPNUM"])
        t_rec = str(header["T_REC"])
        date_obs = pd.Timestamp(header["DATE-OBS"])  # UTC
        name = target.name
        rows.append({
            "harpnum": harp,
            "noaa_ar": int(header["NOAA_AR"]),
            "t_rec": t_rec,
            "timestamp": date_obs,
            "image_path": f"sharp_patches/{name}",
            "height": shape[0],
            "width": shape[1],
            "bytes": target.stat().st_size,
            "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        })
    manifest = pd.DataFrame(rows).sort_values(["timestamp", "harpnum"])
    manifest.to_csv(MANIFEST, index=False)
    print(f"{len(manifest)} patches -> {PATCHES}, manifest -> {MANIFEST}")


if __name__ == "__main__":
    main()
