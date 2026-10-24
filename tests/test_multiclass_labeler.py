import math
from fractions import Fraction

import numpy as np
import pandas as pd
import pytest

import flare_indexer as fidx
from flare_indexer.classifier import FluxConverter
from flare_indexer.events import FlareEvent

flux = FluxConverter().to_flux
labeler = fidx.MulticlassLabeler()

BOUNDARIES = {"A": 1e-8, "B": 1e-7, "C": 1e-6, "M": 1e-5, "X": 1e-4}


def _event(goes_class):
    return FlareEvent(peak_time=pd.Timestamp("2024-01-01"), goes_class=goes_class,
                      start_time=pd.Timestamp("2024-01-01"), active_region=None)


def _multiclass_pipeline():
    return fidx.DatasetBuildingPipeline(reducer=fidx.MaxFluxReducer(), labeler=fidx.MulticlassLabeler())


# ── assign() ─────────────────────────────────────────────────────────────

def test_classes_attribute_lists_labels_in_flux_order():
    assert fidx.MulticlassLabeler.CLASSES == ("NF", "A", "B", "C", "M", "X")


@pytest.mark.parametrize("value", [0.0, 0, -0.0, np.float64(0.0)])
def test_zero_flux_is_no_flare(value):
    assert labeler.assign(value) == "NF"


def test_no_events_through_the_pipeline_is_no_flare():
    assert _multiclass_pipeline().label([]) == "NF"


@pytest.mark.parametrize("goes_class, expected", [
    ("A1.0", "A"), ("A5.5", "A"), ("A9.9", "A"),
    ("B1.0", "B"), ("B9.9", "B"),
    ("C1.0", "C"), ("C9.99", "C"),
    ("M1.0", "M"), ("M9.9", "M"),
    ("X1.0", "X"), ("X9.3", "X"), ("X28", "X"),
])
def test_each_class(goes_class, expected):
    assert labeler.assign(flux(goes_class)) == expected


@pytest.mark.parametrize("letter, boundary", BOUNDARIES.items())
def test_exact_boundary_belongs_to_the_higher_class(letter, boundary):
    assert labeler.assign(boundary) == letter


LOWER_CLASS = {"A": None, "B": "A", "C": "B", "M": "C", "X": "M"}


@pytest.mark.parametrize("letter, boundary", BOUNDARIES.items())
def test_adjacent_floats_around_every_boundary(letter, boundary):
    """Exact boundaries: one representable float below stays in the lower class."""
    above = math.nextafter(boundary, math.inf)
    below = math.nextafter(boundary, 0.0)
    assert labeler.assign(boundary) == labeler.assign(above) == letter
    if LOWER_CLASS[letter] is None:
        with pytest.raises(ValueError, match="below the A-class boundary"):
            labeler.assign(below)
    else:
        assert labeler.assign(below) == LOWER_CLASS[letter]
        assert labeler.assign(boundary * (1 - 5e-10)) == LOWER_CLASS[letter]
        assert labeler.assign(boundary * (1 - 1e-6)) == LOWER_CLASS[letter]


@pytest.mark.parametrize("lower, upper", [
    ("A10.0", "B1.0"), ("B10.0", "C1.0"), ("C10.0", "M1.0"), ("M10.0", "X1.0"),
    ("C100", "X1"), ("M2.3", "m2.30"), ("X10", "X10.000"),
])
def test_equivalent_class_text_gives_identical_flux_and_label(lower, upper):
    assert flux(lower) == flux(upper)          # identical floats, not approximately equal
    assert labeler.assign(flux(lower)) == labeler.assign(flux(upper))


def test_class_boundaries_from_text_are_exactly_the_boundaries():
    for letter, boundary in BOUNDARIES.items():
        assert flux(f"{letter}1.0") == boundary
    assert flux("C10.0") == 1e-05 and labeler.assign(flux("C10.0")) == "M"
    assert flux("C9.99") < 1e-05 and labeler.assign(flux("C9.99")) == "C"


@pytest.mark.parametrize("value", [
    1e-5, math.nextafter(1e-5, 0.0), math.nextafter(1e-5, math.inf), 1e-5 * (1 - 5e-10),
    flux("C10.0"), flux("C9.99"), flux("M1.0"), flux("M10.0"), 0.0,
])
def test_binary_m_threshold_agrees_with_multiclass(value):
    binary = fidx.BinaryThresholdLabeler("M").assign(value)
    assert binary == int(labeler.assign(value) in ("M", "X"))


def test_binary_and_multiclass_agree_on_c10_through_the_pipeline():
    events = [_event("C10.0")]
    assert fidx.BinaryThresholdStrategy("M").label(events) == 1
    assert fidx.FlareClassifier().is_strong("C10.0", "M")
    assert _multiclass_pipeline().label(events) == "M"
    assert fidx.MaxFlareStrategy().label(events) == fidx.MaxFlareStrategy().label([_event("M1.0")]) == 1e-5


def test_m10_and_x1_are_both_x_numerically_not_by_text():
    assert flux("M10.0") == flux("X1.0") == 1e-4
    assert labeler.assign(flux("M10.0")) == labeler.assign(flux("X1.0")) == "X"
    # Text ordering would put "M10.0" before "M9.0" and both before "X1.0".
    assert labeler.assign(flux("M9.0")) == "M"
    assert _multiclass_pipeline().label([_event("M9.0"), _event("M10.0")]) == "X"


def test_label_reflects_the_strongest_flare_not_every_class_present():
    events = [_event("C2.0")] * 10 + [_event("X1.0"), _event("B1.0")]
    assert _multiclass_pipeline().label(events) == "X"


@pytest.mark.parametrize("value", [np.float64(2.3e-5), np.float32(2.3e-5), Fraction(23, 1_000_000)])
def test_numeric_types_are_accepted(value):
    assert labeler.assign(value) == "M"


@pytest.mark.parametrize("value", [5e-9, flux("A0.5"), math.nextafter(1e-8, 0), 1e-300])
def test_positive_flux_below_a_class_is_rejected_not_no_flare(value):
    with pytest.raises(ValueError, match="below the A-class boundary"):
        labeler.assign(value)


@pytest.mark.parametrize("value", [-1e-6, -1e-300, float("-inf")])
def test_negative_flux_is_rejected(value):
    with pytest.raises(ValueError):
        labeler.assign(value)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), np.nan])
def test_non_finite_flux_is_rejected(value):
    with pytest.raises(ValueError, match="non-finite|negative"):
        labeler.assign(value)


@pytest.mark.parametrize("value", ["M2.0", "1e-5", None, True, False, [1e-5], np.bool_(True), complex(1e-5, 0)])
def test_non_numeric_input_is_rejected(value):
    with pytest.raises(TypeError, match="expects a flux number"):
        labeler.assign(value)


# ── End to end through DatasetBuilder ────────────────────────────────────

@pytest.fixture
def catalog(tmp_path):
    path = tmp_path / "catalog.csv"
    pd.DataFrame({
        "start_time": ["2024-01-01 03:00", "2024-01-01 05:00", "2024-01-02 02:00", "2024-01-02 04:00",
                       "2024-01-03 01:00", "2024-01-04 06:00"],
        "peak_time": ["2024-01-01 03:10", "2024-01-01 05:10", "2024-01-02 02:10", "2024-01-02 04:10",
                      "2024-01-03 01:10", "2024-01-04 06:10"],
        "end_time": ["2024-01-01 03:20", "2024-01-01 05:20", "2024-01-02 02:20", "2024-01-02 04:20",
                     "2024-01-03 01:20", "2024-01-04 06:20"],
        "class": ["C3.0", "M10.0", "B2.0", "A4.0", "C9.9", "X1.0"],
        "active_region": [100, 200, 100, 200, 100, 200],
    }).to_csv(path, index=False)
    return path


def _builder(strategy, **kwargs):
    return fidx.DatasetBuilder(prediction_window="12h", strategy=strategy,
                               event_time="start", **kwargs)


@pytest.fixture
def full_disk_index(tmp_path):
    path = tmp_path / "images.csv"
    pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01 00:00", periods=8, freq="12h"),
        "image_path": [f"img{i}.jpg" for i in range(8)],
    }).to_csv(path, index=False)
    return path


def test_single_image_full_disk(catalog, full_disk_index):
    labels = _builder(_multiclass_pipeline(), include_metadata=True).build(full_disk_index, catalog)
    # Windows [t, t+12h): day 1 00h holds C3.0 and M10.0 -> X; day 2 00h holds B2.0 and A4.0 -> B.
    assert labels["label"].tolist() == ["X", "NF", "B", "NF", "C", "NF", "X", "NF"]
    assert labels["image_path"].tolist() == [f"img{i}.jpg" for i in range(8)]


def test_multiclass_agrees_with_binary_and_regression_on_the_same_windows(catalog, full_disk_index):
    multiclass = _builder(_multiclass_pipeline()).build(full_disk_index, catalog)["label"]
    binary = _builder(fidx.BinaryThresholdStrategy("M")).build(full_disk_index, catalog)["label"]
    regression = _builder(fidx.MaxFlareStrategy()).build(full_disk_index, catalog)["label"]

    # Binary and regression results are unchanged by the new labeler.
    assert binary.tolist() == [1, 0, 0, 0, 0, 0, 1, 0]
    assert regression.tolist() == pytest.approx([1e-4, 0.0, 2e-7, 0.0, 9.9e-6, 0.0, 1e-4, 0.0])
    assert [int(c in ("M", "X")) for c in multiclass] == binary.tolist()
    assert [labeler.assign(f) for f in regression] == multiclass.tolist()


def test_row_based_sequences_use_the_final_image(catalog, full_disk_index):
    sequences = _builder(_multiclass_pipeline(), sequence_length=2, stride=2).build(full_disk_index, catalog)
    assert sequences["sequence_end"].astype(str).tolist() == [
        "2024-01-01 12:00:00", "2024-01-02 12:00:00", "2024-01-03 12:00:00", "2024-01-04 12:00:00",
    ]
    assert sequences["label"].tolist() == ["NF", "NF", "NF", "NF"]

    shifted = _builder(_multiclass_pipeline(), sequence_length=2, stride=2).build(
        _drop_first_row(full_disk_index), catalog)
    assert shifted["label"].tolist() == ["B", "C", "X"]


def _drop_first_row(path):
    index = pd.read_csv(path).iloc[1:]
    out = path.with_name("shifted.csv")
    index.to_csv(out, index=False)
    return out


def test_time_based_sequences_use_the_final_image(catalog, full_disk_index):
    sequences, report = _builder(
        _multiclass_pipeline(), cadence="12h", observation_window="24h", sliding_window="12h",
    ).build(full_disk_index, catalog, return_report=True)
    assert report.emitted_sequences == 7
    expected = dict(zip(pd.date_range("2024-01-01 12:00", periods=7, freq="12h"),
                        ["NF", "B", "NF", "C", "NF", "X", "NF"]))
    assert dict(zip(sequences["sequence_end"], sequences["label"])) == expected


@pytest.fixture
def ar_index(tmp_path):
    path = tmp_path / "ar_images.csv"
    times = pd.to_datetime(["2024-01-01 00:00", "2024-01-02 00:00", "2024-01-03 00:00", "2024-01-04 00:00"])
    pd.DataFrame({
        "timestamp": times.repeat(2),
        "active_region": [100, 200] * 4,
        "image_path": [f"{r}_{i}.jpg" for i in range(4) for r in (100, 200)],
    }).to_csv(path, index=False)
    return path


def test_active_region_single_image_isolates_regions(catalog, ar_index):
    labels = _builder(_multiclass_pipeline(), target="active_region", include_metadata=True).build(ar_index, catalog)
    got = {(str(t.date()), r): label for t, r, label in zip(labels["timestamp"], labels["active_region"], labels["label"])}
    assert got == {
        ("2024-01-01", "100"): "C", ("2024-01-01", "200"): "X",   # full disk would say X for both
        ("2024-01-02", "100"): "B", ("2024-01-02", "200"): "A",
        ("2024-01-03", "100"): "C", ("2024-01-03", "200"): "NF",
        ("2024-01-04", "100"): "NF", ("2024-01-04", "200"): "X",
    }
    full_disk = _builder(_multiclass_pipeline()).build(ar_index, catalog)
    assert full_disk["label"].iloc[0] == full_disk["label"].iloc[1] == "X"


def test_active_region_time_based_sequences_isolate_regions(catalog, ar_index):
    sequences = _builder(
        _multiclass_pipeline(), target="active_region",
        cadence="24h", observation_window="48h", sliding_window="24h",
    ).build(ar_index, catalog)
    got = {(str(end.date()), region): label
           for end, region, label in zip(sequences["sequence_end"], sequences["active_region"], sequences["label"])}
    assert got == {
        ("2024-01-02", "100"): "B", ("2024-01-02", "200"): "A",
        ("2024-01-03", "100"): "C", ("2024-01-03", "200"): "NF",
        ("2024-01-04", "100"): "NF", ("2024-01-04", "200"): "X",
    }
    for paths, region in zip(sequences["image_path"], sequences["active_region"]):
        assert all(p.startswith(f"{region}_") for p in paths)


def test_catalog_class_below_a_stops_the_build(tmp_path, full_disk_index):
    path = tmp_path / "sub_a.csv"
    pd.DataFrame({"start_time": ["2024-01-01 03:00"], "peak_time": ["2024-01-01 03:10"],
                  "end_time": ["2024-01-01 03:20"], "class": ["A0.5"], "active_region": [100]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="below the A-class boundary"):
        _builder(_multiclass_pipeline()).build(full_disk_index, path)
