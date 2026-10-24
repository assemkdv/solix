"""
Checks for the real-image PyTorch examples (examples 06, 07 and 10).

The committed provenance files (image manifest, flare catalog) are checked
always. Labeling is checked from the manifest's timestamps alone, which
needs neither images nor PyTorch. Running 06 and 07 needs PyTorch, Pillow
with JPEG 2000 support and the images downloaded by
examples/real_data/fetch_real_data.py; 10 needs PyTorch, astropy and the
SHARP patches. Those tests are skipped otherwise.
"""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

import flare_indexer as fidx
from flare_indexer.builder import DatasetBuilder

REPO = Path(__file__).resolve().parent.parent
EXAMPLES = REPO / "examples"
REAL_DATA = EXAMPLES / "real_data"

pytestmark = pytest.mark.skipif(not REAL_DATA.is_dir(), reason="examples/ not available (installed package)")

if REAL_DATA.is_dir():
    sys.path.insert(0, str(REAL_DATA))
    import hmi_jp2
    import real_examples as real

# Recorded from the executed examples (see examples/real_data/README.md).
# Timestamps are full-precision DATE_OBS values, never rounded.
EXPECTED_IMAGE_LABELS = {
    "2017-09-03 00:59:49.900": 0, "2017-09-03 01:59:49.900": 0, "2017-09-03 02:59:49.900": 0,
    "2017-09-03 03:59:49.900": 0, "2017-09-03 04:59:50.000": 0, "2017-09-03 05:59:50.000": 1,
    "2017-09-03 07:59:50.000": 1, "2017-09-03 09:00:34.900": 1, "2017-09-03 09:59:49.900": 1,
}
EXPECTED_SEQUENCE_ENDS = {"2017-09-03 02:59:49.900": 0, "2017-09-03 03:59:49.900": 0}


def _builder(**sequence):
    return DatasetBuilder(
        prediction_window="24h", strategy=fidx.BinaryThresholdStrategy("M"), target="full_disk",
        include_metadata=not sequence, **sequence,
    )


@pytest.fixture
def index_csv(tmp_path):
    manifest = pd.read_csv(real.MANIFEST)
    available = manifest[manifest["status"] == "ok"]
    path = tmp_path / "index.csv"
    pd.DataFrame({"timestamp": available["timestamp"], "image_path": available["image_path"]}).to_csv(path, index=False)
    return path


# ── Provenance files ─────────────────────────────────────────────────────

def test_manifest_timestamps_are_header_times_not_requested_times():
    manifest = pd.read_csv(real.MANIFEST)
    assert len(manifest) == 10
    for _, row in manifest.iterrows():
        date_obs = pd.Timestamp(row["date_obs"]).tz_convert("UTC").tz_localize(None)
        offset = date_obs - pd.Timestamp(row["requested_time"])
        assert offset.total_seconds() == row["offset_seconds"]
        # Helioviewer's whole-second date is provenance only.
        assert pd.Timestamp(row["helioviewer_date"]) == date_obs.round("s")
        if row["status"] == "ok":
            assert pd.Timestamp(row["timestamp"]) == date_obs
            assert pd.Timestamp(row["timestamp"]) != pd.Timestamp(row["requested_time"])
            assert abs(offset) <= pd.Timedelta("60s")
            assert row["image_path"] == f"images/hmi_magnetogram_{row['helioviewer_id']}.jp2"
            assert (row["instrume"], row["content"]) == ("HMI_FRONT2", "MAGNETOGRAM")
        else:
            assert row["status"] == "missing" and abs(offset) > pd.Timedelta("60s")
            assert pd.isna(row["timestamp"]) and pd.isna(row["image_path"])


def test_catalog_matches_raw_hek_result_and_covers_every_window():
    raw = pd.read_csv(REAL_DATA / "hek_swpc_flares.csv")
    catalog = real.load_catalog()
    provenance = json.loads(real.CATALOG_PROVENANCE.read_text())
    assert len(raw) == len(catalog) == provenance["catalog_rows"]
    assert (raw["frm_name"] == "SWPC").all()
    assert list(catalog["class"]) == list(raw["fl_goescls"].str.strip())
    assert list(catalog["start_time"]) == list(pd.to_datetime(raw["event_starttime"]))

    times = pd.to_datetime(pd.read_csv(real.MANIFEST)["timestamp"]).dropna()
    real.check_catalog_coverage(times, "24h")  # raises SystemExit if not covered
    with pytest.raises(SystemExit):
        real.check_catalog_coverage([pd.Timestamp("2017-09-05 12:00:00")], "24h")


# ── Labels from the real index (no images or PyTorch needed) ─────────────

def test_image_labels_match_independent_check(index_csv):
    labels = _builder().build(index_csv, real.CATALOG)
    assert dict(zip(labels["timestamp"].astype(str), labels["label"])) == EXPECTED_IMAGE_LABELS
    date_obs = real.date_obs_times()
    assert [date_obs[p] for p in labels["image_path"]] == labels["timestamp"].tolist()
    assert labels["image_path"].tolist() == pd.read_csv(index_csv)["image_path"].tolist()

    catalog = real.load_catalog()
    for t, label in zip(labels["timestamp"], labels["label"]):
        assert real.independent_m_label(catalog, t, "24h")[0] == label
    # The first M-class flare after the images: M1.2 starting 2017-09-04 05:36.
    _, strong = real.independent_m_label(catalog, pd.Timestamp("2017-09-03 05:59:50"), "24h")
    assert strong.iloc[0][["class", "start_time"]].tolist() == ["M1.2", pd.Timestamp("2017-09-04 05:36")]
    assert real.independent_m_label(catalog, pd.Timestamp("2017-09-03 04:59:50"), "24h")[0] == 0


def test_sequences_use_final_observation_and_never_substitute_images(index_csv):
    sequences, report = _builder(cadence="1h", observation_window="3h", sliding_window="1h").build(
        index_csv, real.CATALOG, return_report=True,
    )
    assert dict(zip(sequences["sequence_end"].astype(str), sequences["label"])) == EXPECTED_SEQUENCE_ENDS
    assert (report.total_candidate_sequences, report.emitted_sequences, report.skipped_sequences) == (8, 2, 6)
    # The grid starts at 00:59:49.9. The images at 04:59:50.0 and 05:59:50.0
    # are 0.1 s off it, 07:00 was never acquired, and the image near 09:00
    # was observed at 09:00:34.9; none of them is used in place of a grid time.
    assert [str(t) for t in report.missing_timestamps] == [
        "2017-09-03 04:59:49.900000", "2017-09-03 04:59:49.900000", "2017-09-03 04:59:49.900000",
        "2017-09-03 05:59:49.900000", "2017-09-03 06:59:49.900000", "2017-09-03 07:59:49.900000",
    ]

    index = pd.read_csv(index_csv, parse_dates=["timestamp"])
    path_at = dict(zip(index["timestamp"], index["image_path"]))
    for _, row in sequences.iterrows():
        timestamps = row["timestamps"]
        assert timestamps == sorted(timestamps) and timestamps[-1] == row["sequence_end"]
        assert all(b - a == pd.Timedelta("1h") for a, b in zip(timestamps, timestamps[1:]))
        assert row["image_path"] == [path_at[t] for t in timestamps]
        date_obs = real.date_obs_times()
        assert [date_obs[p] for p in row["image_path"]] == timestamps


def test_observation_time_keeps_sub_second_precision():
    header = {"DATE_OBS": "2017-09-03T04:59:49.90Z"}
    assert hmi_jp2.observation_time(header) == pd.Timestamp("2017-09-03 04:59:49.900")
    assert hmi_jp2.matches_api_date(header, "2017-09-03 04:59:50")
    assert not hmi_jp2.matches_api_date(header, "2017-09-03 04:59:49")


def test_tenth_of_a_second_off_grid_is_not_turned_into_an_hourly_step(tmp_path):
    """Regression: an image 0.1 s after a grid point must not fill that grid point."""
    times = pd.to_datetime([
        "2017-09-03 00:59:49.900", "2017-09-03 01:59:49.900", "2017-09-03 02:59:49.900",
        "2017-09-03 03:59:50.000",  # 0.1 s late
        "2017-09-03 04:59:49.900", "2017-09-03 05:59:49.900",
    ])
    index = tmp_path / "index.csv"
    pd.DataFrame({"timestamp": times, "image_path": [f"img{i}.jp2" for i in range(6)]}).to_csv(index, index=False)
    assert pd.read_csv(index, parse_dates=["timestamp"])["timestamp"].tolist() == list(times)

    sequences, report = _builder(cadence="1h", observation_window="2h", sliding_window="1h").build(
        index, real.CATALOG, return_report=True,
    )
    assert report.inferred_source_cadence == pd.Timedelta("1h")
    assert [str(t) for t in report.missing_timestamps] == ["2017-09-03 03:59:49.900000"] * 2
    assert sequences["sequence_end"].astype(str).tolist() == [
        "2017-09-03 01:59:49.900", "2017-09-03 02:59:49.900", "2017-09-03 05:59:49.900",
    ]
    assert all("img3.jp2" not in paths for paths in sequences["image_path"])
    for timestamps in sequences["timestamps"]:
        assert all(b - a == pd.Timedelta("1h") for a, b in zip(timestamps, timestamps[1:]))


def test_sequence_csv_round_trip(index_csv, tmp_path):
    sequences = _builder(cadence="1h", observation_window="3h", sliding_window="1h").build(index_csv, real.CATALOG)
    path = tmp_path / "sequences.csv"
    real.save_sequences(sequences, path)
    reloaded = real.load_sequences(path)
    pd.testing.assert_frame_equal(reloaded, sequences, check_dtype=False)
    assert isinstance(reloaded["image_path"].iloc[0], list)
    assert isinstance(reloaded["timestamps"].iloc[0][0], pd.Timestamp)


# ── Real images and PyTorch ──────────────────────────────────────────────

def _images_available():
    manifest = pd.read_csv(real.MANIFEST)
    available = manifest[manifest["status"] == "ok"]
    return all(
        (REAL_DATA / p).exists() and hashlib.sha256((REAL_DATA / p).read_bytes()).hexdigest() == sha
        for p, sha in zip(available["image_path"], available["sha256"])
    )


needs_images = pytest.mark.skipif(
    not REAL_DATA.is_dir() or not _images_available(),
    reason="real images not downloaded (python examples/real_data/fetch_real_data.py)",
)


@needs_images
def test_decoded_images_are_real_and_match_header_times():
    pytest.importorskip("PIL")
    manifest = pd.read_csv(real.MANIFEST)
    available = manifest[manifest["status"] == "ok"]
    arrays = []
    for path, timestamp in zip(available["image_path"], available["timestamp"]):
        file = REAL_DATA / path
        assert hmi_jp2.observation_time(hmi_jp2.read_jp2_header(file)) == pd.Timestamp(timestamp)
        array = hmi_jp2.decode_magnetogram(file)
        assert array.shape == (1, 256, 256) and str(array.dtype) == "float32"
        assert 0.0 <= array.min() and array.max() <= 1.0 and array.std() > 0.1
        arrays.append(array)
    assert all((a != b).any() for a, b in zip(arrays, arrays[1:]))


@needs_images
@pytest.mark.parametrize("script, expected", [
    ("06_pytorch_images.py", ["images [4, 1, 256, 256]", "images [1, 1, 256, 256]",
                                        "9 images: 4 positive, 5 negative; all checks passed"]),
    ("07_pytorch_sequences.py", ["images [2, 3, 1, 256, 256]", "candidates 8, emitted 2, skipped 6",
                                           "2 sequences: 0 positive, 2 negative; all checks passed"]),
])
def test_pytorch_example_runs(script, expected, tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("PIL")
    completed = subprocess.run(
        [sys.executable, str(EXAMPLES / script), "--output", str(tmp_path)],
        capture_output=True, text=True, timeout=300,
    )
    assert completed.returncode == 0, completed.stderr
    for line in expected:
        assert line in completed.stdout


# ── Active-region patches (examples 10 and 11) ───────────────────────────

# Recorded from the executed example: HARP 7115 = NOAA 12673 gets the M1.2
# that starts 2017-09-04 05:36 in its window from 05:58:42.5 on; HARP 7120 =
# NOAA 12675 has no flares.
EXPECTED_PATCH_LABELS = {
    (pd.Timestamp("2017-09-03 05:58:42.500"), "12673"): 1, (pd.Timestamp("2017-09-03 08:58:42.400"), "12673"): 1,
}


def _patches_available():
    manifest = pd.read_csv(real.SHARP_MANIFEST)
    return all(
        (REAL_DATA / p).exists() and hashlib.sha256((REAL_DATA / p).read_bytes()).hexdigest() == sha
        for p, sha in zip(manifest["image_path"], manifest["sha256"])
    )


needs_patches = pytest.mark.skipif(not REAL_DATA.is_dir() or not _patches_available(),
                                   reason="SHARP patches not present (python examples/real_data/fetch_sharp_patches.py)")


def test_patch_manifest_lists_two_regions_at_the_same_times():
    manifest = pd.read_csv(real.SHARP_MANIFEST)
    assert len(manifest) == 16
    assert dict(zip(manifest["harpnum"], manifest["noaa_ar"])) == {7115: 12673, 7120: 12675}
    times = manifest.groupby("harpnum")["timestamp"].apply(list)
    assert times[7115] == times[7120]
    # T_REC is the TAI record time; the timestamp is DATE-OBS in UTC, before it.
    assert all(pd.Timestamp(ts) < pd.Timestamp(t.replace("_TAI", "").replace(".", "-", 2).replace("_", " "))
               for ts, t in zip(manifest["timestamp"], manifest["t_rec"]))


def test_patch_labels_count_only_each_regions_own_flares(tmp_path):
    index_path = tmp_path / "patch_index.csv"
    index = real.build_patch_index(index_path) if _patches_available() else None
    if index is None:
        manifest = pd.read_csv(real.SHARP_MANIFEST)
        pd.DataFrame({"timestamp": manifest["timestamp"], "active_region": manifest["noaa_ar"],
                      "image_path": manifest["image_path"]}).to_csv(index_path, index=False)
    labels = fidx.ActiveRegionSingleImage("24h", fidx.BinaryThresholdStrategy("M")).build(index_path, real.CATALOG)
    got = {(t, r): label for t, r, label in zip(labels["timestamp"], labels["active_region"], labels["label"])}
    assert {key: value for key, value in got.items() if value} == EXPECTED_PATCH_LABELS
    assert len(got) == 16
    catalog = real.load_catalog()
    for (t, region), label in got.items():
        assert real.independent_m_label(catalog, t, "24h", region)[0] == label


@needs_patches
def test_decoded_patches_are_real_and_match_header_times():
    pytest.importorskip("astropy")
    pytest.importorskip("PIL")
    import sharp_fits

    manifest = pd.read_csv(real.SHARP_MANIFEST)
    arrays = []
    for path, timestamp, harp, noaa in zip(manifest["image_path"], manifest["timestamp"],
                                           manifest["harpnum"], manifest["noaa_ar"]):
        header, field = sharp_fits.read_patch(REAL_DATA / path)
        assert (header["HARPNUM"], header["NOAA_AR"]) == (harp, noaa)
        assert sharp_fits.observation_time(header) == pd.Timestamp(timestamp)
        patch = sharp_fits.load_patch(REAL_DATA / path, timestamp)
        assert patch.shape == (1, 128, 128) and str(patch.dtype) == "float32"
        assert 0.0 <= patch.min() and patch.max() <= 1.0 and patch.std() > 0.01
        arrays.append(patch)
    assert all((a != b).any() for a, b in zip(arrays, arrays[1:]))
    with pytest.raises(ValueError, match="header time"):
        sharp_fits.load_patch(REAL_DATA / manifest["image_path"][0], "2017-09-03 00:00:00")


@needs_patches
def test_pytorch_active_region_example_runs(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("astropy")
    completed = subprocess.run(
        [sys.executable, str(EXAMPLES / "10_pytorch_active_region.py"), "--output", str(tmp_path)],
        capture_output=True, text=True, timeout=300,
    )
    assert completed.returncode == 0, completed.stderr
    assert "batch 3: patches [4, 1, 128, 128] torch.float32  regions ['12673', '12675', '12673', '12675']  " \
           "labels [1, 0, 1, 0]" in completed.stdout
    assert "16 patches: 2 positive, 14 negative; all checks passed" in completed.stdout
