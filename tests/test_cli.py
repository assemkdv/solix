"""
The `flare-indexer` command (flare_indexer.cli) must give the same result as
the Python API it wraps, refuse what does not fit, and never leave a partial
output behind.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

import flare_indexer as fidx
from flare_indexer.cli import main

FIXTURES = Path(__file__).resolve().parent.parent / "docs" / "fixtures"
pytestmark = pytest.mark.skipif(not FIXTURES.is_dir(), reason="docs/fixtures not available (installed package)")
IMAGES = FIXTURES / "images.csv"          # 5 full-disk images, 3 h apart (one 9 h gap), with image_path
AR_IMAGES = FIXTURES / "ar_images.csv"    # patches of two regions at shared times, one blank region
CATALOG = FIXTURES / "catalog.csv"        # 4 flares, one without a region


def run(*argv):
    return main([str(a) for a in argv])


def no_stray_files(directory: Path, *expected: Path):
    return sorted(p.name for p in directory.iterdir()) == sorted(p.name for p in expected)


# --- help, version ---------------------------------------------------------

def test_help_shows_commands_and_defaults(capsys):
    with pytest.raises(SystemExit) as excinfo:
        run("--help")
    assert excinfo.value.code == 0
    assert "label" in capsys.readouterr().out

    with pytest.raises(SystemExit) as excinfo:
        run("label", "--help")
    assert excinfo.value.code == 0
    out = " ".join(capsys.readouterr().out.split())   # argparse wraps long lines
    for text in ("(default: 24h)", "(default: peak)", "(default: left_closed)", "(default: full_disk)",
                 "(default: binary)", "--overwrite", "--sequence-length", "--observation-window", "--report"):
        assert text in out


def test_version(capsys):
    with pytest.raises(SystemExit) as excinfo:
        run("--version")
    assert excinfo.value.code == 0
    assert capsys.readouterr().out.startswith("flare-indexer ")


def test_module_is_runnable_as_a_script():
    completed = subprocess.run(
        [sys.executable, "-m", "flare_indexer.cli", "--version"], capture_output=True, text=True, timeout=60,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.startswith("flare-indexer ")


# --- agreement with the Python API, single images --------------------------

SINGLE_IMAGE_CASES = [
    pytest.param([], dict(prediction_window="24h", strategy=fidx.BinaryThresholdStrategy()), id="binary-defaults"),
    pytest.param(
        ["--threshold", "C4.2", "--event-time", "start", "--interval-mode", "right_closed", "--window", "6h"],
        dict(prediction_window="6h", strategy=fidx.BinaryThresholdStrategy("C4.2"),
             event_time="start", interval_mode="right_closed"),
        id="binary-C4.2-start-right_closed-6h",
    ),
    pytest.param(["--labeling", "max-flux"], dict(prediction_window="24h", strategy=fidx.MaxFlareStrategy()), id="max-flux"),
    pytest.param(
        ["--labeling", "multiclass"],
        dict(prediction_window="24h",
             strategy=fidx.DatasetBuildingPipeline(fidx.MaxFluxReducer(), fidx.MulticlassLabeler())),
        id="multiclass",
    ),
    pytest.param(
        ["--labeling", "cumulative", "--window", "23h59min59s", "--event-time", "start", "--interval-mode", "right_closed"],
        dict(prediction_window="23h59min59s", event_time="start", interval_mode="right_closed",
             strategy=fidx.DatasetBuildingPipeline(fidx.CumulativeClassIndexReducer(), fidx.RegressionLabeler())),
        id="cumulative-raw",
    ),
    pytest.param(
        ["--labeling", "cumulative", "--threshold", "10"],
        dict(prediction_window="24h",
             strategy=fidx.DatasetBuildingPipeline(fidx.CumulativeClassIndexReducer(), fidx.NumericThresholdLabeler(10))),
        id="cumulative-threshold-10",
    ),
]


@pytest.mark.parametrize("cli_args, kwargs", SINGLE_IMAGE_CASES)
def test_single_image_output_matches_api(tmp_path, cli_args, kwargs):
    output = tmp_path / "labels.csv"
    assert run("label", IMAGES, CATALOG, "--output", output, *cli_args) == 0
    expected = fidx.DatasetBuilder(include_metadata=True, **kwargs).build(IMAGES, CATALOG)
    got = pd.read_csv(output, parse_dates=["timestamp"])
    assert list(got.columns) == list(expected.columns) == ["timestamp", "label", "image_path"]
    assert got["timestamp"].tolist() == expected["timestamp"].tolist()
    assert got["image_path"].tolist() == expected["image_path"].tolist()
    pd.testing.assert_series_equal(got["label"], expected["label"], check_dtype=False)


def test_cumulative_raw_and_thresholded_values(tmp_path):
    raw = tmp_path / "raw.csv"
    binary = tmp_path / "binary.csv"
    common = ["--labeling", "cumulative", "--window", "6h", "--event-time", "start"]
    assert run("label", IMAGES, CATALOG, "--output", raw, *common) == 0
    assert run("label", IMAGES, CATALOG, "--output", binary, *common, "--threshold", "20") == 0
    raw_labels = pd.read_csv(raw)["label"]
    binary_labels = pd.read_csv(binary)["label"]
    assert raw_labels.dtype.kind == "f" and raw_labels.max() > 0      # the raw index is a float
    assert binary_labels.tolist() == [int(v >= 20) for v in raw_labels]


def test_class_counts_and_presence_cells_are_json_dicts(tmp_path):
    for labeling, labeler in (("class-counts", fidx.ClassCountLabeler()), ("class-presence", fidx.ClassPresenceLabeler())):
        output = tmp_path / f"{labeling}.csv"
        assert run("label", IMAGES, CATALOG, "--output", output, "--labeling", labeling) == 0
        expected = fidx.DatasetBuilder(
            "24h", fidx.DatasetBuildingPipeline(fidx.ClassCountReducer(), labeler), include_metadata=True,
        ).build(IMAGES, CATALOG)
        got = pd.read_csv(output, converters={"label": json.loads})
        assert got["label"].tolist() == expected["label"].tolist()
        assert all(list(d) == ["A", "B", "C", "M", "X"] for d in got["label"])


def test_active_region_single_images_keep_identity_and_warn(tmp_path, capsys):
    output = tmp_path / "ar.csv"
    assert run("label", AR_IMAGES, CATALOG, "--output", output, "--target", "active_region") == 0
    err = capsys.readouterr().err
    assert "flare-indexer: warning:" in err and "missing active_region" in err
    expected = fidx.DatasetBuilder(
        "24h", fidx.BinaryThresholdStrategy(), target="active_region", include_metadata=True,
    ).build(AR_IMAGES, CATALOG)
    got = pd.read_csv(output, dtype={"active_region": str})
    assert list(got.columns) == list(expected.columns)
    assert got["label"].tolist() == expected["label"].tolist()
    assert got["image_path"].tolist() == expected["image_path"].tolist()
    assert got["active_region"].fillna("").tolist() == [r or "" for r in expected["active_region"]]


def test_missing_active_region_policy_is_passed_through(tmp_path, capsys):
    output = tmp_path / "ar.csv"
    assert run("label", AR_IMAGES, CATALOG, "--output", output, "--target", "active_region",
               "--missing-active-region", "drop") == 0
    expected = fidx.DatasetBuilder(
        "24h", fidx.BinaryThresholdStrategy(), target="active_region", missing_active_region="drop",
        include_metadata=True,
    ).build(AR_IMAGES, CATALOG)
    assert len(pd.read_csv(output)) == len(expected)
    assert "missing active_region" not in capsys.readouterr().err


# --- sequences and JSON round trips ---------------------------------------

def test_row_based_sequences_match_api_and_json_cells_round_trip(tmp_path):
    output = tmp_path / "seq.csv"
    assert run("label", IMAGES, CATALOG, "--output", output, "--sequence-length", "2", "--stride", "1") == 0
    expected = fidx.DatasetBuilder("24h", fidx.BinaryThresholdStrategy(), sequence_length=2, stride=1).build(IMAGES, CATALOG)
    got = pd.read_csv(
        output, parse_dates=["sequence_start", "sequence_end"],
        converters={"timestamps": json.loads, "image_path": json.loads},
    )
    assert list(got.columns) == list(expected.columns)
    assert got["label"].tolist() == expected["label"].tolist()
    assert got["n_images"].tolist() == expected["n_images"].tolist()
    assert got["image_path"].tolist() == expected["image_path"].tolist()
    assert [[pd.Timestamp(t) for t in ts] for ts in got["timestamps"]] == [list(ts) for ts in expected["timestamps"]]
    # The raw cell is valid JSON text, readable with json alone.
    raw = pd.read_csv(output, dtype=str)["timestamps"].iloc[0]
    assert json.loads(raw) == [str(t) for t in expected["timestamps"].iloc[0]]


def test_cadence_minutes_skips_gaps_like_the_api(tmp_path):
    output = tmp_path / "seq.csv"
    assert run("label", IMAGES, CATALOG, "--output", output, "--sequence-length", "2", "--cadence-minutes", "180") == 0
    expected = fidx.DatasetBuilder("24h", fidx.BinaryThresholdStrategy(), sequence_length=2, cadence_minutes=180).build(IMAGES, CATALOG)
    assert len(pd.read_csv(output)) == len(expected)


def test_time_based_sequences_match_api_and_write_report(tmp_path):
    output = tmp_path / "seq.csv"
    report = tmp_path / "report.json"
    assert run("label", IMAGES, CATALOG, "--output", output, "--cadence", "3h", "--observation-window", "6h",
               "--sliding-window", "3h", "--report", report) == 0
    expected, expected_report = fidx.DatasetBuilder(
        "24h", fidx.BinaryThresholdStrategy(), cadence="3h", observation_window="6h", sliding_window="3h",
    ).build(IMAGES, CATALOG, return_report=True)
    got = pd.read_csv(output, converters={"timestamps": json.loads, "image_path": json.loads})
    assert list(got.columns) == list(expected.columns)
    assert got["label"].tolist() == expected["label"].tolist()
    assert got["image_path"].tolist() == expected["image_path"].tolist()
    written = json.loads(report.read_text())
    assert written["emitted_sequences"] == expected_report.emitted_sequences
    assert written["skipped_sequences"] == expected_report.skipped_sequences
    assert written["total_candidate_sequences"] == expected_report.total_candidate_sequences
    assert pd.Timedelta(written["inferred_source_cadence"]) == expected_report.inferred_source_cadence
    assert pd.Timestamp(written["effective_start"]) == expected_report.effective_start


def test_active_region_time_based_sequences_match_api(tmp_path):
    output = tmp_path / "seq.csv"
    assert run("label", AR_IMAGES, CATALOG, "--output", output, "--target", "active_region",
               "--cadence", "3h", "--observation-window", "6h", "--sliding-window", "3h") == 0
    expected = fidx.DatasetBuilder(
        "24h", fidx.BinaryThresholdStrategy(), target="active_region",
        cadence="3h", observation_window="6h", sliding_window="3h",
    ).build(AR_IMAGES, CATALOG)
    got = pd.read_csv(output, dtype={"active_region": str}, converters={"timestamps": json.loads})
    assert list(got.columns) == list(expected.columns)
    assert got["label"].tolist() == expected["label"].tolist()
    assert got["active_region"].fillna("").tolist() == [r or "" for r in expected["active_region"]]


# --- refusals ---------------------------------------------------------------

@pytest.mark.parametrize("extra, message", [
    (["--window", "24"], "has no unit"),
    (["--cadence", "3600", "--observation-window", "2h", "--sliding-window", "1h"], "has no unit"),
    (["--labeling", "max-flux", "--threshold", "M"], "--threshold is not used by --labeling max-flux"),
    (["--labeling", "class-counts", "--threshold", "10"], "--threshold is not used"),
    (["--labeling", "cumulative", "--threshold", "M"], "must be a number"),
    (["--labeling", "binary", "--threshold", "10"], "Invalid GOES threshold"),
    (["--sequence-length", "2", "--cadence", "3h"], "cannot combine row-based options (--sequence-length) with time-based options (--cadence)"),
    (["--stride", "2"], "need --sequence-length"),
    (["--cadence-minutes", "60"], "need --sequence-length"),
    (["--report", "r.json"], "--report is only produced in time-based sequence mode"),
    (["--missing-active-region", "drop"], "--missing-active-region only applies with --target active_region"),
    (["--cadence", "3h"], "Time-based sequence mode also requires: --observation-window, --sliding-window"),
    (["--cadence", "3h", "--observation-window", "4h", "--sliding-window", "3h"], "--observation-window (4h) must be an exact multiple of --cadence (3h)"),
])
def test_incompatible_or_invalid_options_are_refused(tmp_path, capsys, extra, message):
    output = tmp_path / "labels.csv"
    assert run("label", IMAGES, CATALOG, "--output", output, *extra) == 1
    err = capsys.readouterr().err
    assert err.startswith("flare-indexer: error: ") and message in err
    assert no_stray_files(tmp_path)


@pytest.mark.parametrize("argv", [
    ["label", str(IMAGES), str(CATALOG), "--output", "x.csv", "--labeling", "nonsense"],
    ["label", str(IMAGES), str(CATALOG), "--output", "x.csv", "--sequence-length", "0"],
    ["label", str(IMAGES), str(CATALOG), "--output", "x.csv", "--sequence-length", "2.5"],
    ["label", str(IMAGES), str(CATALOG), "--output", "x.csv", "--cadence-minutes", "-5"],
    ["label", str(IMAGES), str(CATALOG)],                        # --output is required
    ["label", str(IMAGES)],
    ["frobnicate"],
    [],
])
def test_usage_errors_exit_with_status_2(argv, capsys, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit) as excinfo:
        main(argv)
    assert excinfo.value.code == 2
    assert capsys.readouterr().err
    assert no_stray_files(tmp_path)


def test_missing_input_files_are_reported(tmp_path, capsys):
    output = tmp_path / "labels.csv"
    assert run("label", tmp_path / "nope.csv", CATALOG, "--output", output) == 1
    assert "input file not found" in capsys.readouterr().err
    assert run("label", IMAGES, tmp_path / "nope.csv", "--output", output) == 1
    assert "input file not found" in capsys.readouterr().err
    assert no_stray_files(tmp_path)


def test_bad_input_data_leaves_no_output(tmp_path, capsys):
    bad_catalog = tmp_path / "catalog.csv"
    bad_catalog.write_text("start_time,peak_time,end_time,class,active_region\n20240101100000,20240101101000,20240101103000,M1.0,1\n")
    output = tmp_path / "labels.csv"
    assert run("label", IMAGES, bad_catalog, "--output", output) == 1
    err = capsys.readouterr().err
    assert "flare-indexer: error: Flare catalog at" in err and "not date-time text" in err
    assert no_stray_files(tmp_path, bad_catalog)


def test_existing_output_is_not_replaced_without_overwrite(tmp_path, capsys):
    output = tmp_path / "labels.csv"
    assert run("label", IMAGES, CATALOG, "--output", output) == 0
    first = output.read_text()
    assert run("label", IMAGES, CATALOG, "--output", output, "--labeling", "max-flux") == 1
    assert "output file exists" in capsys.readouterr().err
    assert output.read_text() == first
    assert run("label", IMAGES, CATALOG, "--output", output, "--labeling", "max-flux", "--overwrite") == 0
    assert output.read_text() != first
    assert no_stray_files(tmp_path, output)


def test_report_path_is_protected_too(tmp_path, capsys):
    output = tmp_path / "seq.csv"
    report = tmp_path / "report.json"
    report.write_text("{}")
    time_based = ["--cadence", "3h", "--observation-window", "6h", "--sliding-window", "3h"]
    assert run("label", IMAGES, CATALOG, "--output", output, *time_based, "--report", report) == 1
    assert "output file exists" in capsys.readouterr().err
    assert report.read_text() == "{}" and not output.exists()
    assert run("label", IMAGES, CATALOG, "--output", output, *time_based, "--report", output) == 1
    assert "must be different files" in capsys.readouterr().err
    assert run("label", IMAGES, CATALOG, "--output", output, *time_based, "--report", report, "--overwrite") == 0
    assert json.loads(report.read_text())["emitted_sequences"] >= 0


def test_write_failure_leaves_no_partial_file(tmp_path, capsys, monkeypatch):
    output = tmp_path / "out" / "labels.csv"
    import flare_indexer.cli as cli

    def failing_to_csv(self, handle, *args, **kwargs):
        handle.write("partial")
        raise OSError("disk full (simulated)")

    monkeypatch.setattr(pd.DataFrame, "to_csv", failing_to_csv)
    assert run("label", IMAGES, CATALOG, "--output", output) == 1
    assert "disk full" in capsys.readouterr().err
    assert not output.exists() and no_stray_files(output.parent)


def test_documented_example_command_runs(tmp_path):
    # The command shown in the researcher manual, section 18.
    output = tmp_path / "labels.csv"
    assert run("label", IMAGES, CATALOG, "--output", output, "--labeling", "cumulative", "--threshold", "10",
               "--window", "23h59min59s", "--event-time", "start", "--interval-mode", "right_closed") == 0
    assert set(pd.read_csv(output)["label"]) <= {0, 1}
