"""
Checks for scripts/catalog_corrections.py: the documented peak_time
corrections the validation scripts apply to the lab's GOES catalog.

Most tests use a small catalog built from the seven original records. The
last test needs the real cached catalog (scripts/.cache, not public) and is
skipped without it.
"""

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

from flare_indexer import DatasetBuilder, BinaryThresholdStrategy, adapt_goes_catalog
from flare_indexer.events import EventMatcher

REPO = Path(__file__).resolve().parent.parent
LAB_CATALOG = REPO / "scripts" / ".cache" / "goes_flares_integrated.csv"

spec = importlib.util.spec_from_file_location("catalog_corrections", REPO / "scripts" / "catalog_corrections.py")
corrections = importlib.util.module_from_spec(spec)
spec.loader.exec_module(corrections)

COLUMNS = ["start_time", "peak_time", "end_time", "goes_class", "noaa_active_region"]
EXTRA_ROWS = [
    ("2010-05-01 01:34:00", "2010-05-01 01:39:00", "2010-05-01 01:43:00", "C5.7", 11067.0),
    ("2010-06-11 23:56:00", "2010-06-12 00:02:00", "2010-06-12 00:04:00", "B6.3", None),  # a correct roll-over
]


def _original_rows():
    return [(c.start_time, c.peak_time, c.end_time, c.goes_class, float(c.noaa_active_region))
            for c in corrections.CORRECTIONS]


def _write(path, rows):
    frame = pd.DataFrame(rows, columns=COLUMNS)
    frame.insert(0, "Unnamed: 0", range(len(frame)))   # the upstream file has a nameless index column
    frame["other_column"] = "kept"
    frame.to_csv(path, index=False)
    return path


def test_seven_corrections_match_the_documented_records():
    assert len(corrections.CORRECTIONS) == 7
    for c in corrections.CORRECTIONS:
        start, peak, end = (pd.Timestamp(t) for t in (c.start_time, c.peak_time, c.end_time))
        assert peak == start.normalize(), "every original peak is 00:00 on the start date"
        assert end.normalize() == start.normalize() + pd.Timedelta("1D"), "every flare ends on the next day"
        if c.corrected_peak_time is corrections.MISSING:
            assert "////" in c.ngdc_line
        else:
            assert c.ngdc_line.endswith(" 0000")
            assert pd.Timestamp(c.corrected_peak_time) == start.normalize() + pd.Timedelta("1D")
    assert sum(c.corrected_peak_time is corrections.MISSING for c in corrections.CORRECTIONS) == 1


def test_corrections_are_applied_and_reported(tmp_path):
    source = _write(tmp_path / "goes.csv", EXTRA_ROWS[:1] + _original_rows() + EXTRA_ROWS[1:])
    before = source.read_bytes()

    output, report = corrections.apply_catalog_corrections(source, tmp_path / "out" / "corrected.csv")

    assert source.read_bytes() == before, "the source file is never modified"
    assert report["source_is_the_known_catalog"] is False and len(report["corrections"]) == 7
    assert [c["data_row"] for c in report["corrections"]] == list(range(2, 9))

    corrected = pd.read_csv(output)
    assert list(corrected.columns) == ["Unnamed: 0"] + COLUMNS + ["other_column"]
    assert corrected["other_column"].eq("kept").all()
    # Untouched rows are byte-for-byte the same values.
    pd.testing.assert_frame_equal(corrected.iloc[[0, 8]].reset_index(drop=True),
                                  pd.read_csv(source).iloc[[0, 8]].reset_index(drop=True))
    peaks = corrected["peak_time"].tolist()
    assert pd.isna(peaks[1]), "2011-04-11 C2.5: NOAA records no maximum -> missing"
    assert peaks[2:8] == ["2012-04-06 00:00:00", "2012-06-07 00:00:00", "2012-07-31 00:00:00",
                          "2012-10-14 00:00:00", "2014-05-26 00:00:00", "2017-03-27 00:00:00"]
    for row in report["corrections"]:
        assert row["ngdc_file"].startswith("goes-xrs-report_") and row["reason"]


def test_corrected_catalog_passes_the_time_order_check_and_labels_by_start_only(tmp_path):
    source = _write(tmp_path / "goes.csv", _original_rows())
    with pytest.raises(ValueError, match="start <= peak <= end"):
        adapt_goes_catalog(source, tmp_path / "adapted_uncorrected.csv")
        EventMatcher(tmp_path / "adapted_uncorrected.csv")

    output, _ = corrections.apply_catalog_corrections(source, tmp_path / "corrected.csv")
    adapted = adapt_goes_catalog(output, tmp_path / "adapted.csv")
    assert adapted["peak_time"].isna().tolist() == [True] + [False] * 6
    matcher = EventMatcher(tmp_path / "adapted.csv")          # time order now valid

    # The 2011-04-11 flare (no peak) is found by start time only.
    t = pd.Timestamp("2011-04-11 12:00:00")
    assert [f.goes_class for f in matcher.query(t, 24, event_time="start")] == ["C2.5"]
    assert matcher.query(t, 24, event_time="peak") == []
    # A corrected flare is found by both, in the window that contains each time.
    t = pd.Timestamp("2012-10-13 23:00:00")
    assert [f.goes_class for f in matcher.query(t, 2, event_time="start")] == ["C1.5"]
    assert [f.goes_class for f in matcher.query(t, 2, event_time="peak")] == ["C1.5"]
    assert matcher.query(pd.Timestamp("2012-10-13 00:00:00"), 1, event_time="peak") == []


@pytest.mark.parametrize("change", ["already_corrected", "class", "duplicate"])
def test_records_that_do_not_match_exactly_are_refused(tmp_path, change):
    rows = _original_rows()
    if change == "already_corrected":
        rows[4] = rows[4][:1] + ("2012-10-14 00:00:00",) + rows[4][2:]
    elif change == "class":
        rows[4] = rows[4][:3] + ("C1.6",) + rows[4][4:]
    else:
        rows.append(rows[4])
    source = _write(tmp_path / "goes.csv", rows)

    with pytest.raises(ValueError, match="expected exactly one row") as excinfo:
        corrections.apply_catalog_corrections(source, tmp_path / "corrected.csv")
    assert "2012-10-13 23:46:00" in str(excinfo.value)
    assert not (tmp_path / "corrected.csv").exists()


def test_start_time_labels_are_unchanged_by_the_corrections(tmp_path):
    source = _write(tmp_path / "goes.csv", EXTRA_ROWS + _original_rows())
    index = tmp_path / "index.csv"
    pd.DataFrame({"timestamp": pd.to_datetime(
        ["2011-04-11 00:00", "2011-04-11 23:00", "2012-10-13 00:00", "2012-10-13 23:00", "2017-03-27 00:00"]
    )}).to_csv(index, index=False)
    output, _ = corrections.apply_catalog_corrections(source, tmp_path / "corrected.csv")
    adapt_goes_catalog(output, tmp_path / "adapted.csv")
    # The uncorrected catalog fails flare_indexer's time-order check, so the
    # reference is the lab's rule computed directly: any flare with
    # start_time in (t, t + 24h].
    raw = pd.read_csv(source)
    starts = pd.to_datetime(raw["start_time"])
    expected = [int(((starts > t) & (starts <= t + pd.Timedelta("24h")) & raw["goes_class"].str.startswith("C")).any())
                for t in pd.to_datetime(pd.read_csv(index)["timestamp"])]
    result = DatasetBuilder(prediction_window="24h", strategy=BinaryThresholdStrategy("C"), target="full_disk",
                            event_time="start", interval_mode="right_closed").build(index, tmp_path / "adapted.csv")
    assert result["label"].tolist() == expected == [1, 1, 1, 1, 0]


@pytest.mark.skipif(not LAB_CATALOG.exists(), reason="lab catalog scripts/.cache/goes_flares_integrated.csv not present")
def test_real_lab_catalog_gets_exactly_the_documented_corrections(tmp_path):
    output, report = corrections.apply_catalog_corrections(LAB_CATALOG, tmp_path / "corrected.csv")
    assert report["source_is_the_known_catalog"], "the cached upstream file changed"
    assert [c["data_row"] for c in report["corrections"]] == [1592, 3534, 3940, 4357, 4824, 8328, 13145]
    adapted = adapt_goes_catalog(output, tmp_path / "adapted.csv")
    assert len(adapted) == 14401 and int(adapted["peak_time"].isna().sum()) == 1
    EventMatcher(tmp_path / "adapted.csv")      # no time-order error anywhere in the corrected catalog
