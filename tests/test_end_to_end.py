from pathlib import Path

import pandas as pd
import pytest

from flare_indexer.builder import DatasetBuilder
from flare_indexer.events import EventMatcher
from flare_indexer.strategies import BinaryThresholdStrategy, MaxFlareStrategy

FIXTURES_DIR = Path(__file__).parent / "fixtures"
CATALOG_PATH = FIXTURES_DIR / "catalog.csv"
IMAGE_INDEX_PATH = FIXTURES_DIR / "image_index.csv"

# tests/fixtures/catalog.csv has four flares:
#   C3.0 peak 2024-02-01T12:00:00  (flux=3e-6)
#   X1.0 peak 2024-02-02T00:00:00  (flux=1e-4)  <- exactly on a 24h window end
#   C1.0 peak 2024-02-05T06:00:00  (flux=1e-6)
#   M4.0 peak 2024-02-05T18:00:00  (flux=4e-5)
#
# tests/fixtures/image_index.csv has three timestamps, each with a
# prediction_window of 24 hours:
#   2024-02-01T00:00:00 -> window [2024-02-01T00:00, 2024-02-02T00:00)
#       Only C3.0 is inside. X1.0 peaks exactly at the window end and must
#       be excluded by the half-open interval.
#   2024-02-05T00:00:00 -> window [2024-02-05T00:00, 2024-02-06T00:00)
#       Both C1.0 and M4.0 are inside; M4.0 qualifies the binary threshold.
#   2024-02-10T00:00:00 -> window [2024-02-10T00:00, 2024-02-11T00:00)
#       No flares fall inside; the window is empty.


def test_end_to_end_binary_threshold_strategy():
    builder = DatasetBuilder(prediction_window=24, strategy=BinaryThresholdStrategy())
    result = builder.build(IMAGE_INDEX_PATH, CATALOG_PATH)

    assert list(result.columns) == ["timestamp", "label"]
    assert result["timestamp"].tolist() == [
        pd.Timestamp("2024-02-01T00:00:00"),
        pd.Timestamp("2024-02-05T00:00:00"),
        pd.Timestamp("2024-02-10T00:00:00"),
    ]
    assert result["label"].tolist() == [0, 1, 0]


def test_end_to_end_max_flare_strategy():
    builder = DatasetBuilder(prediction_window=24, strategy=MaxFlareStrategy())
    result = builder.build(IMAGE_INDEX_PATH, CATALOG_PATH)

    # Row 1: max is C3.0's 3e-6 — X1.0's 1e-4 must be excluded since its
    # peak lands exactly on the window boundary. If the boundary were
    # mistakenly inclusive, this would be 1e-4 instead.
    # Row 2: max of C1.0 (1e-6) and M4.0 (4e-5) is 4e-5.
    # Row 3: no flares in the window -> 0.0.
    assert result["label"].tolist() == [3e-6, 4e-5, 0.0]


def test_end_to_end_sequences_against_real_fixture_catalog(tmp_path):
    # Uses the same real-format (date + HHMM) tests/fixtures/catalog.csv as
    # the tests above, but drives it through sequence labeling rather than
    # single-image labeling.
    index_path = tmp_path / "sequence_index.csv"
    pd.DataFrame({
        "timestamp": pd.to_datetime([
            "2024-02-01T09:00:00",
            "2024-02-01T10:00:00",
            "2024-02-01T11:00:00",
            "2024-02-01T12:00:00",
        ]),
    }).to_csv(index_path, index=False)

    builder = DatasetBuilder(
        prediction_window=1,
        strategy=BinaryThresholdStrategy(),
        sequence_length=2,
        stride=1,
        cadence_minutes=60,
    )
    result = builder.build(index_path, CATALOG_PATH)

    assert list(result.columns) == ["sequence_start", "sequence_end", "timestamps", "n_images", "label"]
    assert result["sequence_end"].tolist() == [
        pd.Timestamp("2024-02-01T10:00:00"),
        pd.Timestamp("2024-02-01T11:00:00"),
        pd.Timestamp("2024-02-01T12:00:00"),
    ]
    # Only the last sequence's 1-hour window ([12:00, 13:00)) catches a
    # flare (C3.0, peak exactly at 12:00) -- and it's C-class, so the
    # M-threshold binary label is still 0 for all three sequences.
    assert result["label"].tolist() == [0, 0, 0]

    max_result = DatasetBuilder(
        prediction_window=1,
        strategy=MaxFlareStrategy(),
        sequence_length=2,
        stride=1,
        cadence_minutes=60,
    ).build(index_path, CATALOG_PATH)
    assert max_result["label"].tolist() == [0.0, 0.0, 3e-6]


def test_end_to_end_active_region_against_real_fixture_catalog(tmp_path):
    # tests/fixtures/catalog.csv has two active regions: 11111 (C3.0, X1.0
    # on 2024-02-01/02) and 22222 (C1.0, M4.0 on 2024-02-05). Using
    # threshold="C" so C-class flares count as qualifying.
    index_path = tmp_path / "ar_index.csv"
    pd.DataFrame({
        "timestamp": pd.to_datetime([
            "2024-02-01T00:00:00",  # region 11111, correct -> catches C3.0
            "2024-02-05T00:00:00",  # region 11111, wrong that day -> nothing
            "2024-02-05T00:00:00",  # region 22222, correct -> catches C1.0/M4.0
        ]),
        "active_region": ["11111", "11111", "22222"],
    }).to_csv(index_path, index=False)

    builder = DatasetBuilder(
        prediction_window=24,
        strategy=BinaryThresholdStrategy(threshold="C"),
        target="active_region",
    )
    result = builder.build(index_path, CATALOG_PATH)

    assert list(result.columns) == ["timestamp", "label"]
    assert result["label"].tolist() == [1, 0, 1]


def test_end_to_end_active_region_sequences_against_real_fixture_catalog(tmp_path):
    # tests/fixtures/catalog.csv has region 11111 (C3.0 peak 2024-02-01T12:00,
    # X1.0 peak 2024-02-02T00:00) and region 22222 (C1.0/M4.0 on 2024-02-05).
    # Using threshold="C" so C-class flares count as qualifying.
    index_path = tmp_path / "ar_sequence_index.csv"
    pd.DataFrame({
        "timestamp": pd.to_datetime([
            "2024-02-01T11:00:00",
            "2024-02-01T11:30:00",
            "2024-02-01T12:00:00",
            "2024-02-05T09:00:00",
            "2024-02-05T09:30:00",
            "2024-02-05T10:00:00",
        ]),
        "active_region": ["11111", "11111", "11111", "22222", "22222", "22222"],
    }).to_csv(index_path, index=False)

    builder = DatasetBuilder(
        prediction_window=1,
        strategy=BinaryThresholdStrategy(threshold="C"),
        target="active_region",
        sequence_length=3,
        stride=3,
        cadence_minutes=30,
    )
    result = builder.build(index_path, CATALOG_PATH)

    assert list(result.columns) == [
        "sequence_start", "sequence_end", "timestamps", "n_images", "active_region", "label"
    ]
    assert result["sequence_end"].tolist() == [
        pd.Timestamp("2024-02-01T12:00:00"),
        pd.Timestamp("2024-02-05T10:00:00"),
    ]
    # First sequence ends exactly at region 11111's C3.0 peak -> label 1.
    # Second sequence's 1-hour window from 10:00 catches no region-22222
    # flare (C1.0 peaked at 06:00, M4.0 peaks later at 18:00) -> label 0.
    assert result["label"].tolist() == [1, 0]
    assert result["active_region"].tolist() == ["11111", "22222"]


# The two synthetic catalogs hold INVENTED events, not solar observations;
# see tests/fixtures/README.md.
SYNTHETIC_HHMM_CATALOG_PATH = FIXTURES_DIR / "synthetic_hhmm_catalog.csv"
SYNTHETIC_FULL_TIMESTAMP_CATALOG_PATH = FIXTURES_DIR / "synthetic_full_timestamp_catalog.csv"


def _write_index(path, timestamps):
    pd.DataFrame({"timestamp": pd.to_datetime(timestamps)}).to_csv(path, index=False)
    return path


def _times(matcher):
    return matcher._catalog[["goes_class", "start_time", "peak_time", "end_time"]].to_dict("records")


def test_end_to_end_synthetic_hhmm_catalog(tmp_path):
    # Every HHMM time takes the calendar date from the row's `date` column.
    assert _times(EventMatcher(SYNTHETIC_HHMM_CATALOG_PATH)) == [
        {"goes_class": "M2.0",
         "start_time": pd.Timestamp("2030-01-10 09:40"),
         "peak_time": pd.Timestamp("2030-01-10 09:52"),
         "end_time": pd.Timestamp("2030-01-10 10:05")},
        # Equal start, peak and end are allowed.
        {"goes_class": "C3.0",
         "start_time": pd.Timestamp("2030-01-12 12:00"),
         "peak_time": pd.Timestamp("2030-01-12 12:00"),
         "end_time": pd.Timestamp("2030-01-12 12:00")},
    ]

    index_path = _write_index(tmp_path / "index.csv",
                              ["2030-01-10T00:00:00", "2030-01-12T00:00:00", "2030-01-13T00:00:00"])

    # Defaults: event_time="peak", interval_mode="left_closed" -> [t, t + 24h).
    #   2030-01-10 00:00: M2.0 (peak 09:52) -> 1, max flux 2e-5
    #   2030-01-12 00:00: only C3.0 (peak 12:00), below M -> 0, max flux 3e-6
    #   2030-01-13 00:00: nothing -> 0, 0.0
    binary = DatasetBuilder(prediction_window=24, strategy=BinaryThresholdStrategy("M")) \
        .build(index_path, SYNTHETIC_HHMM_CATALOG_PATH)
    assert list(binary.columns) == ["timestamp", "label"]
    assert binary["timestamp"].tolist() == [
        pd.Timestamp("2030-01-10T00:00:00"),
        pd.Timestamp("2030-01-12T00:00:00"),
        pd.Timestamp("2030-01-13T00:00:00"),
    ]
    assert binary["label"].tolist() == [1, 0, 0]

    flux = DatasetBuilder(prediction_window=24, strategy=MaxFlareStrategy()) \
        .build(index_path, SYNTHETIC_HHMM_CATALOG_PATH)
    assert flux["label"].tolist() == pytest.approx([2e-5, 3e-6, 0.0])


def test_end_to_end_full_timestamp_midnight_crossing_event(tmp_path):
    assert _times(EventMatcher(SYNTHETIC_FULL_TIMESTAMP_CATALOG_PATH)) == [
        {"goes_class": "X1.5",
         "start_time": pd.Timestamp("2030-01-11 23:50"),
         "peak_time": pd.Timestamp("2030-01-12 00:05"),
         "end_time": pd.Timestamp("2030-01-12 00:20")},
    ]

    index_path = _write_index(tmp_path / "index.csv",
                              ["2030-01-11T00:00:00", "2030-01-12T00:00:00", "2030-01-13T00:00:00"])

    # Windows [t, t + 24h) (default interval_mode):
    #   2030-01-11 00:00: start 01-11 23:50 inside; peak 01-12 00:05 outside
    #   2030-01-12 00:00: start before t; peak 01-12 00:05 inside
    #   2030-01-13 00:00: neither
    labels = {}
    for event_time in ("start", "peak"):
        result = DatasetBuilder(prediction_window=24, strategy=BinaryThresholdStrategy("M"), event_time=event_time) \
            .build(index_path, SYNTHETIC_FULL_TIMESTAMP_CATALOG_PATH)
        assert result["timestamp"].tolist() == [
            pd.Timestamp("2030-01-11T00:00:00"),
            pd.Timestamp("2030-01-12T00:00:00"),
            pd.Timestamp("2030-01-13T00:00:00"),
        ]
        labels[event_time] = result["label"].tolist()
    assert labels == {"start": [1, 0, 0], "peak": [0, 1, 0]}
