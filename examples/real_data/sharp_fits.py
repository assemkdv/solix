"""
Reading the SHARP active-region patches used by examples 10 and 11.

Each file is one hmi.sharp_cea_720s magnetogram (line-of-sight field in
Gauss). Patches differ in size, so they are resized to one size for
batching. Reading needs astropy; resizing needs Pillow.
"""

from pathlib import Path

import numpy as np
import pandas as pd

PATCH_SIZE = 128            # every patch is resized to PATCH_SIZE x PATCH_SIZE
FIELD_RANGE = (-256, 255)   # Gauss; the clip used in arpatch-sfp (data_process/mag-process)


def read_patch(path: Path) -> tuple[dict, np.ndarray]:
    """The header and the magnetogram (Gauss, float) of one SHARP file."""
    from astropy.io import fits

    with fits.open(path) as hdus:
        hdu = next(h for h in hdus if h.data is not None)
        return dict(hdu.header), np.asarray(hdu.data, dtype=np.float64)


def observation_time(header: dict) -> pd.Timestamp:
    """DATE-OBS, the observation time in UTC (T_REC is the TAI record time)."""
    return pd.Timestamp(header["DATE-OBS"])


def load_patch(path: Path, expected_time) -> np.ndarray:
    """
    Check the header time, then return float32 [1, PATCH_SIZE, PATCH_SIZE]:

    1. missing pixels (NaN) are set to 0 G;
    2. the field is clipped to FIELD_RANGE and scaled to [0, 1];
    3. the patch is resized (bilinear) to PATCH_SIZE x PATCH_SIZE.
    """
    from PIL import Image

    header, field = read_patch(path)
    observed = observation_time(header)
    if observed != pd.Timestamp(expected_time):
        raise ValueError(f"{path}: header time {observed} != index time {expected_time}")
    low, high = FIELD_RANGE
    scaled = (np.clip(np.nan_to_num(field, nan=0.0), low, high) - low) / (high - low)
    resized = Image.fromarray(scaled.astype(np.float32)).resize((PATCH_SIZE, PATCH_SIZE), Image.BILINEAR)
    return np.array(resized, dtype=np.float32)[np.newaxis]   # a writable copy
