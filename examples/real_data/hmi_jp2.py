"""
Reading the Helioviewer HMI JPEG 2000 files used by examples 07, 08, 10 and 11.

Kept apart from fetch_real_data.py so the examples can read and decode the
cached files without any network code. Decoding needs Pillow built with
OpenJPEG (the standard Pillow wheels are); Pillow is imported only when an
image is actually decoded.
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd

HEADER_KEYS = ("DATE_OBS", "T_OBS", "T_REC", "TELESCOP", "INSTRUME", "CONTENT", "WAVELNTH", "QUALITY", "CROTA2")

# Preprocessing shared by both examples (see decode_magnetogram).
SOURCE_SIZE = 4096
REDUCE = 4                      # decode at 1 / 2**4 of full resolution
IMAGE_SIZE = SOURCE_SIZE >> REDUCE  # 256


def read_jp2_header(path: Path) -> dict:
    """Read selected FITS keywords from the XML box Helioviewer embeds in each JP2."""
    data = Path(path).read_bytes()
    start, end = data.find(b"<fits"), data.find(b"</fits>")
    if start < 0 or end < 0:
        raise ValueError(f"{path} has no embedded FITS header")
    xml = data[start:end].decode("utf-8", "replace")
    header = {}
    for key in HEADER_KEYS:
        match = re.search(rf"<{key}>(.*?)</{key}>", xml, flags=re.DOTALL)
        header[key] = match.group(1).strip() if match else None
    return header


def observation_time(header: dict) -> pd.Timestamp:
    """
    The image timestamp used for indexing, sequence construction and label
    reference times: DATE_OBS (UTC) at full precision, e.g.
    2017-09-03 00:59:49.900. It is never rounded, so an image observed 0.1 s
    off an hourly grid point stays off it.
    """
    return pd.Timestamp(header["DATE_OBS"]).tz_convert("UTC").tz_localize(None)


def matches_api_date(header: dict, api_date) -> bool:
    """
    Provenance check only: Helioviewer reports `date` rounded to whole
    seconds, so compare it with a rounded copy of DATE_OBS. The rounded
    value is never used as an image timestamp.
    """
    return observation_time(header).round("s") == pd.Timestamp(api_date)


def decode_magnetogram(path: Path) -> np.ndarray:
    """
    Decode one JP2 into a float32 array of shape [1, 256, 256].

    1. The 4096 x 4096 8-bit image is decoded at JPEG 2000 resolution
       level 4 (OpenJPEG's wavelet reduction), giving 256 x 256.
    2. Values are divided by 255, giving [0, 1]. They are Helioviewer's
       8-bit display scaling of the magnetogram, not calibrated Gauss.
    3. A channel axis is added (C = 1).
    No rotation, cropping or limb masking is applied (the files are
    already north-up: CROTA2 = 0).
    """
    from PIL import Image

    with Image.open(path) as image:
        if image.format != "JPEG2000" or image.mode != "L" or image.size != (SOURCE_SIZE, SOURCE_SIZE):
            raise ValueError(f"{path}: expected an 8-bit {SOURCE_SIZE}x{SOURCE_SIZE} JPEG 2000, "
                             f"got {image.format} {image.mode} {image.size}")
        image.reduce = REDUCE
        image.load()
        pixels = np.asarray(image, dtype=np.float32)
    if pixels.shape != (IMAGE_SIZE, IMAGE_SIZE):
        raise ValueError(f"{path}: decoded shape {pixels.shape}, expected {(IMAGE_SIZE, IMAGE_SIZE)}")
    return (pixels / 255.0)[np.newaxis]
