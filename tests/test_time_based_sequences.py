import pandas as pd
import pytest

from flare_indexer.builder import DatasetBuilder, SequenceBuildReport
from flare_indexer.strategies import BinaryThresholdStrategy, MaxFlareStrategy


def _write_index(tmp_path, timestamps, extra_columns=None, active_regions=None, name="index.csv"):
    data = {"timestamp": timestamps}
    if active_regions is not None:
        data["active_region"] = active_regions
    if extra_columns:
        data.update(extra_columns)
    path = tmp_path / name
    pd.DataFrame(data).to_csv(path, index=False)
    return path


def _hourly(start, count):
    return list(pd.date_range(start, periods=count, freq="1h"))


def _empty_catalog(tmp_path):
    path = tmp_path / "catalog.csv"
    path.write_text("start_time,peak_time,end_time,class,active_region\n")
    return path


def _catalog_with_flare(tmp_path, date, start, peak, end, cls, active_region):
    # start/peak/end are "HHMM" clock times on `date`, written as full timestamps.
    def stamp(hhmm):
        return f"{date} {hhmm[:2]}:{hhmm[2:]}:00"
    path = tmp_path / "catalog.csv"
    path.write_text(
        "start_time,peak_time,end_time,class,active_region\n"
        f"{stamp(start)},{stamp(peak)},{stamp(end)},{cls},{active_region}\n"
    )
    return path


# ── Parameter validation ─────────────────────────────────────────────────


def test_missing_cadence_raises():
    with pytest.raises(ValueError) as excinfo:
        DatasetBuilder(
            prediction_window=24, strategy=BinaryThresholdStrategy(),
            observation_window="6h", sliding_window="3h",
        )
    assert "cadence" in str(excinfo.value)


def test_missing_observation_window_raises():
    with pytest.raises(ValueError) as excinfo:
        DatasetBuilder(
            prediction_window=24, strategy=BinaryThresholdStrategy(),
            cadence="1h", sliding_window="3h",
        )
    assert "observation_window" in str(excinfo.value)


def test_missing_sliding_window_raises():
    with pytest.raises(ValueError) as excinfo:
        DatasetBuilder(
            prediction_window=24, strategy=BinaryThresholdStrategy(),
            cadence="1h", observation_window="6h",
        )
    assert "sliding_window" in str(excinfo.value)


def test_zero_duration_cadence_raises():
    with pytest.raises(ValueError) as excinfo:
        DatasetBuilder(
            prediction_window=24, strategy=BinaryThresholdStrategy(),
            cadence="0h", observation_window="6h", sliding_window="3h",
        )
    assert "cadence" in str(excinfo.value)


def test_negative_duration_observation_window_raises():
    with pytest.raises(ValueError) as excinfo:
        DatasetBuilder(
            prediction_window=24, strategy=BinaryThresholdStrategy(),
            cadence="1h", observation_window="-6h", sliding_window="3h",
        )
    assert "observation_window" in str(excinfo.value)


def test_invalid_duration_string_raises():
    with pytest.raises(ValueError) as excinfo:
        DatasetBuilder(
            prediction_window=24, strategy=BinaryThresholdStrategy(),
            cadence="banana", observation_window="6h", sliding_window="3h",
        )
    assert "cadence" in str(excinfo.value)


def test_observation_window_not_divisible_by_cadence_raises():
    with pytest.raises(ValueError) as excinfo:
        DatasetBuilder(
            prediction_window=24, strategy=BinaryThresholdStrategy(),
            cadence="1h", observation_window="90min", sliding_window="3h",
        )
    assert "observation_window" in str(excinfo.value)


def test_sliding_window_smaller_than_cadence(tmp_path):
    # 12-minute images, hourly sequences of 3 images, a new one every 12 minutes.
    timestamps = list(pd.date_range("2024-03-01", periods=16, freq="12min"))
    builder = DatasetBuilder(
        prediction_window=24, strategy=BinaryThresholdStrategy(),
        cadence="1h", observation_window="3h", sliding_window="12min",
    )
    result = builder.build(_write_index(tmp_path, timestamps), _empty_catalog(tmp_path))
    assert result["sequence_start"].tolist() == timestamps[:6]
    assert result["timestamps"].iloc[1] == [timestamps[1], timestamps[6], timestamps[11]]


@pytest.mark.parametrize("target", ["full_disk", "active_region"])
def test_sliding_window_between_images_raises(tmp_path, target):
    index = _write_index(tmp_path, _hourly("2024-03-01", 12), active_regions=[13600] * 12)
    builder = DatasetBuilder(
        prediction_window=24, strategy=BinaryThresholdStrategy(), target=target,
        cadence="2h", observation_window="4h", sliding_window="90min",
    )
    with pytest.raises(ValueError, match="sliding_window 0 days 01:30:00 is not an exact multiple"):
        builder.build(index, _empty_catalog(tmp_path))


def test_invalid_start_value_raises():
    with pytest.raises(ValueError) as excinfo:
        DatasetBuilder(
            prediction_window=24, strategy=BinaryThresholdStrategy(),
            start="not-a-date", cadence="1h", observation_window="6h", sliding_window="3h",
        )
    assert "start" in str(excinfo.value)


# ── Core time-based behavior ─────────────────────────────────────────────


def test_six_images_per_sequence_and_expected_starts_ends(tmp_path):
    # 10 hourly images: 00:00 .. 09:00. cadence=1h, observation_window=6h
    # (6 images), sliding_window=3h.
    timestamps = _hourly("2024-01-01T00:00:00", 10)
    index_path = _write_index(tmp_path, timestamps)
    catalog_path = _empty_catalog(tmp_path)

    builder = DatasetBuilder(
        prediction_window=24, strategy=BinaryThresholdStrategy(),
        cadence="1h", observation_window="6h", sliding_window="3h",
    )
    result = builder.build(index_path, catalog_path)

    assert list(result.columns) == ["sequence_start", "sequence_end", "timestamps", "n_images", "label"]
    assert len(result) == 2
    assert result["sequence_start"].tolist() == [
        pd.Timestamp("2024-01-01T00:00:00"), pd.Timestamp("2024-01-01T03:00:00"),
    ]
    assert result["sequence_end"].tolist() == [
        pd.Timestamp("2024-01-01T05:00:00"), pd.Timestamp("2024-01-01T08:00:00"),
    ]
    assert result["n_images"].tolist() == [6, 6]
    assert result.iloc[0]["timestamps"] == _hourly("2024-01-01T00:00:00", 6)
    assert result.iloc[1]["timestamps"] == _hourly("2024-01-01T03:00:00", 6)


def test_observation_window_is_half_open_last_image_at_t_plus_5h(tmp_path):
    timestamps = _hourly("2024-01-01T00:00:00", 6)
    index_path = _write_index(tmp_path, timestamps)
    catalog_path = _empty_catalog(tmp_path)

    builder = DatasetBuilder(
        prediction_window=24, strategy=BinaryThresholdStrategy(),
        cadence="1h", observation_window="6h", sliding_window="6h",
    )
    result = builder.build(index_path, catalog_path)

    assert len(result) == 1
    # [t, t+6h) -> images at t..t+5h, NOT t+6h.
    assert result.iloc[0]["sequence_end"] == pd.Timestamp("2024-01-01T05:00:00")
    assert result.iloc[0]["n_images"] == 6
    assert pd.Timestamp("2024-01-01T06:00:00") not in result.iloc[0]["timestamps"]


def test_label_uses_final_image_timestamp_not_sequence_start(tmp_path):
    timestamps = _hourly("2024-01-01T00:00:00", 6)
    index_path = _write_index(tmp_path, timestamps)
    # Flare peaks one minute after the final image's timestamp (05:00); a
    # window measured from sequence_start (00:00) with a short
    # prediction_window would miss it, but one from sequence_end catches it.
    catalog_path = _catalog_with_flare(tmp_path, "2024-01-01", "0501", "0501", "0505", "M1.0", "4456")

    builder = DatasetBuilder(
        prediction_window=0.1,  # 6 minutes
        strategy=BinaryThresholdStrategy(),
        cadence="1h", observation_window="6h", sliding_window="6h",
    )
    result = builder.build(index_path, catalog_path)

    assert len(result) == 1
    assert result.iloc[0]["label"] == 1


def test_insufficient_data_returns_empty_dataframe_with_correct_schema(tmp_path):
    timestamps = _hourly("2024-01-01T00:00:00", 3)  # only 00,01,02 -- not enough for a 6-image sequence
    index_path = _write_index(tmp_path, timestamps)
    catalog_path = _empty_catalog(tmp_path)

    builder = DatasetBuilder(
        prediction_window=24, strategy=BinaryThresholdStrategy(),
        cadence="1h", observation_window="6h", sliding_window="3h",
    )
    result = builder.build(index_path, catalog_path)

    assert list(result.columns) == ["sequence_start", "sequence_end", "timestamps", "n_images", "label"]
    assert len(result) == 0


def test_missing_required_timestamp_skips_sequence_later_one_still_emitted(tmp_path):
    # 12 hourly slots 00:00..11:00, but 04:00 is missing.
    timestamps = [t for t in _hourly("2024-01-01T00:00:00", 12) if t != pd.Timestamp("2024-01-01T04:00:00")]
    index_path = _write_index(tmp_path, timestamps)
    catalog_path = _empty_catalog(tmp_path)

    builder = DatasetBuilder(
        prediction_window=24, strategy=BinaryThresholdStrategy(),
        cadence="1h", observation_window="6h", sliding_window="3h",
    )
    result, report = builder.build(index_path, catalog_path, return_report=True)

    # start=00:00 needs [00..05], missing 04 -> skip.
    # start=03:00 needs [03..08], missing 04 -> skip.
    # start=06:00 needs [06..11], all present -> emit.
    assert report.total_candidate_sequences == 3
    assert report.emitted_sequences == 1
    assert report.skipped_sequences == 2
    assert report.skipped_sequence_starts == [
        pd.Timestamp("2024-01-01T00:00:00"), pd.Timestamp("2024-01-01T03:00:00"),
    ]
    assert report.missing_timestamps == [pd.Timestamp("2024-01-01T04:00:00")] * 2
    assert len(result) == 1
    assert result.iloc[0]["sequence_start"] == pd.Timestamp("2024-01-01T06:00:00")
    assert result.iloc[0]["sequence_end"] == pd.Timestamp("2024-01-01T11:00:00")


def test_no_interpolation_or_substitution_for_missing_timestamp(tmp_path):
    # Confirms a skipped candidate never appears with a repeated or
    # substituted image in place of the missing one.
    timestamps = [t for t in _hourly("2024-01-01T00:00:00", 7) if t != pd.Timestamp("2024-01-01T03:00:00")]
    index_path = _write_index(tmp_path, timestamps)
    catalog_path = _empty_catalog(tmp_path)

    builder = DatasetBuilder(
        prediction_window=24, strategy=BinaryThresholdStrategy(),
        cadence="1h", observation_window="6h", sliding_window="6h",
    )
    result = builder.build(index_path, catalog_path)

    assert len(result) == 0  # the only candidate (start=00:00) needs the missing 03:00


# ── Compatible downsampling ──────────────────────────────────────────────


def test_downsampling_thirty_minute_source_to_hourly_cadence(tmp_path):
    timestamps = list(pd.date_range("2024-01-01T00:00:00", periods=9, freq="30min"))  # 00:00 .. 04:00
    index_path = _write_index(tmp_path, timestamps)
    catalog_path = _empty_catalog(tmp_path)

    builder = DatasetBuilder(
        prediction_window=24, strategy=BinaryThresholdStrategy(),
        cadence="1h", observation_window="4h", sliding_window="4h",
    )
    result = builder.build(index_path, catalog_path)

    assert len(result) == 1
    assert result.iloc[0]["timestamps"] == [
        pd.Timestamp("2024-01-01T00:00:00"), pd.Timestamp("2024-01-01T01:00:00"),
        pd.Timestamp("2024-01-01T02:00:00"), pd.Timestamp("2024-01-01T03:00:00"),
    ]
    # The 30-minute-offset rows must be excluded, not included.
    for half_hour in ("00:30:00", "01:30:00", "02:30:00"):
        assert pd.Timestamp(f"2024-01-01T{half_hour}") not in result.iloc[0]["timestamps"]


# ── start semantics ───────────────────────────────────────────────────────


def test_start_omitted_uses_earliest_timestamp(tmp_path):
    timestamps = _hourly("2024-01-01T00:00:00", 10)
    index_path = _write_index(tmp_path, timestamps)
    catalog_path = _empty_catalog(tmp_path)

    builder = DatasetBuilder(
        prediction_window=24, strategy=BinaryThresholdStrategy(),
        cadence="1h", observation_window="6h", sliding_window="3h",
    )
    _, report = builder.build(index_path, catalog_path, return_report=True)

    assert report.effective_start == pd.Timestamp("2024-01-01T00:00:00")


def test_explicit_start_present_excludes_earlier_images(tmp_path):
    timestamps = _hourly("2024-01-01T00:00:00", 10)  # 00:00 .. 09:00
    index_path = _write_index(tmp_path, timestamps)
    catalog_path = _empty_catalog(tmp_path)

    builder = DatasetBuilder(
        prediction_window=24, strategy=BinaryThresholdStrategy(),
        start="2024-01-01T02:00:00", cadence="1h", observation_window="6h", sliding_window="3h",
    )
    result, report = builder.build(index_path, catalog_path, return_report=True)

    assert report.effective_start == pd.Timestamp("2024-01-01T02:00:00")
    assert result["sequence_start"].min() == pd.Timestamp("2024-01-01T02:00:00")
    # No sequence timestamp list may contain anything before start.
    for ts_list in result["timestamps"]:
        assert all(t >= pd.Timestamp("2024-01-01T02:00:00") for t in ts_list)


def test_start_missing_exact_timestamp_raises(tmp_path):
    timestamps = _hourly("2024-01-01T00:00:00", 10)
    index_path = _write_index(tmp_path, timestamps)
    catalog_path = _empty_catalog(tmp_path)

    builder = DatasetBuilder(
        prediction_window=24, strategy=BinaryThresholdStrategy(),
        start="2024-01-01T02:30:00",  # not an exact hourly timestamp
        cadence="1h", observation_window="6h", sliding_window="3h",
    )
    with pytest.raises(ValueError) as excinfo:
        builder.build(index_path, catalog_path)

    assert "start" in str(excinfo.value)


def test_start_before_dataset_raises(tmp_path):
    timestamps = _hourly("2024-01-01T00:00:00", 10)
    index_path = _write_index(tmp_path, timestamps)
    catalog_path = _empty_catalog(tmp_path)

    builder = DatasetBuilder(
        prediction_window=24, strategy=BinaryThresholdStrategy(),
        start="2023-12-31T00:00:00", cadence="1h", observation_window="6h", sliding_window="3h",
    )
    with pytest.raises(ValueError) as excinfo:
        builder.build(index_path, catalog_path)

    assert "start" in str(excinfo.value)


def test_start_after_dataset_raises(tmp_path):
    timestamps = _hourly("2024-01-01T00:00:00", 10)
    index_path = _write_index(tmp_path, timestamps)
    catalog_path = _empty_catalog(tmp_path)

    builder = DatasetBuilder(
        prediction_window=24, strategy=BinaryThresholdStrategy(),
        start="2024-06-01T00:00:00", cadence="1h", observation_window="6h", sliding_window="3h",
    )
    with pytest.raises(ValueError) as excinfo:
        builder.build(index_path, catalog_path)

    assert "start" in str(excinfo.value)


# ── Source cadence inference and validation ──────────────────────────────


def test_hourly_source_rejects_one_minute_cadence(tmp_path):
    timestamps = _hourly("2024-01-01T00:00:00", 10)
    index_path = _write_index(tmp_path, timestamps)
    catalog_path = _empty_catalog(tmp_path)

    builder = DatasetBuilder(
        prediction_window=24, strategy=BinaryThresholdStrategy(),
        cadence="1min", observation_window="6min", sliding_window="6min",
    )
    with pytest.raises(ValueError) as excinfo:
        builder.build(index_path, catalog_path)
    assert "cadence" in str(excinfo.value)


def test_hourly_source_accepts_three_hour_cadence(tmp_path):
    timestamps = _hourly("2024-01-01T00:00:00", 25)  # 00:00 day1 .. 00:00 day2
    index_path = _write_index(tmp_path, timestamps)
    catalog_path = _empty_catalog(tmp_path)

    builder = DatasetBuilder(
        prediction_window=24, strategy=BinaryThresholdStrategy(),
        cadence="3h", observation_window="6h", sliding_window="6h",
    )
    result, report = builder.build(index_path, catalog_path, return_report=True)
    assert report.inferred_source_cadence == pd.Timedelta(hours=1)
    assert len(result) > 0
    assert result.iloc[0]["timestamps"] == [
        pd.Timestamp("2024-01-01T00:00:00"), pd.Timestamp("2024-01-01T03:00:00"),
    ]


def test_hourly_source_rejects_ninety_minute_cadence(tmp_path):
    timestamps = _hourly("2024-01-01T00:00:00", 10)
    index_path = _write_index(tmp_path, timestamps)
    catalog_path = _empty_catalog(tmp_path)

    builder = DatasetBuilder(
        prediction_window=24, strategy=BinaryThresholdStrategy(),
        cadence="90min", observation_window="90min", sliding_window="90min",
    )
    with pytest.raises(ValueError) as excinfo:
        builder.build(index_path, catalog_path)
    assert "cadence" in str(excinfo.value)


def test_occasional_gaps_do_not_corrupt_base_cadence_inference(tmp_path):
    # Mostly hourly, but with a 2h and a 3h gap thrown in -- the mode of
    # positive consecutive diffs must still be 1h, not the largest gap.
    base = pd.Timestamp("2024-01-01T00:00:00")
    timestamps = [base + pd.Timedelta(hours=h) for h in [0, 1, 2, 3, 5, 6, 7, 10, 11, 12, 13, 14]]
    # gaps: 1,1,1,2,1,1,3,1,1,1,1 -- mode is 1h despite the 2h/3h gaps.
    index_path = _write_index(tmp_path, timestamps)
    catalog_path = _empty_catalog(tmp_path)

    builder = DatasetBuilder(
        prediction_window=24, strategy=BinaryThresholdStrategy(),
        cadence="1h", observation_window="3h", sliding_window="3h",
    )
    result, report = builder.build(index_path, catalog_path, return_report=True)
    assert report.inferred_source_cadence == pd.Timedelta(hours=1)


def test_fewer_than_two_unique_timestamps_raises(tmp_path):
    index_path = _write_index(tmp_path, [pd.Timestamp("2024-01-01T00:00:00")])
    catalog_path = _empty_catalog(tmp_path)

    builder = DatasetBuilder(
        prediction_window=24, strategy=BinaryThresholdStrategy(),
        cadence="1h", observation_window="6h", sliding_window="3h",
    )
    with pytest.raises(ValueError) as excinfo:
        builder.build(index_path, catalog_path)
    assert "cadence" in str(excinfo.value).lower()


# ── SequenceBuildReport ───────────────────────────────────────────────────


def test_return_report_default_false_returns_dataframe_only(tmp_path):
    timestamps = _hourly("2024-01-01T00:00:00", 10)
    index_path = _write_index(tmp_path, timestamps)
    catalog_path = _empty_catalog(tmp_path)

    builder = DatasetBuilder(
        prediction_window=24, strategy=BinaryThresholdStrategy(),
        cadence="1h", observation_window="6h", sliding_window="3h",
    )
    result = builder.build(index_path, catalog_path)
    assert isinstance(result, pd.DataFrame)


def test_return_report_true_returns_dataframe_and_report(tmp_path):
    timestamps = _hourly("2024-01-01T00:00:00", 10)
    index_path = _write_index(tmp_path, timestamps)
    catalog_path = _empty_catalog(tmp_path)

    builder = DatasetBuilder(
        prediction_window=24, strategy=BinaryThresholdStrategy(),
        cadence="1h", observation_window="6h", sliding_window="3h",
    )
    result, report = builder.build(index_path, catalog_path, return_report=True)

    assert isinstance(result, pd.DataFrame)
    assert isinstance(report, SequenceBuildReport)
    assert report.requested_cadence == pd.Timedelta(hours=1)
    assert report.observation_window == pd.Timedelta(hours=6)
    assert report.sliding_window == pd.Timedelta(hours=3)
    assert report.total_candidate_sequences == report.emitted_sequences + report.skipped_sequences

def test_return_report_true_in_single_image_mode_raises(tmp_path):
    catalog_path = _empty_catalog(tmp_path)
    index_path = _write_index(tmp_path, _hourly("2024-01-01T00:00:00", 3))

    builder = DatasetBuilder(prediction_window=24, strategy=BinaryThresholdStrategy())
    with pytest.raises(ValueError) as excinfo:
        builder.build(index_path, catalog_path, return_report=True)
    assert "return_report" in str(excinfo.value)


# ── Single-image mode is the default ──────────────────────────────────────

def test_default_is_single_image_mode(tmp_path):
    timestamps = _hourly("2024-01-01T00:00:00", 2)
    index_path = _write_index(tmp_path, timestamps)
    catalog_path = _empty_catalog(tmp_path)

    builder = DatasetBuilder(prediction_window=24, strategy=BinaryThresholdStrategy())
    assert builder.time_based is False

    result = builder.build(index_path, catalog_path)
    assert list(result.columns) == ["timestamp", "label"]


# ── event_time / window boundary / MaxFlareStrategy compatibility ──────────


def test_time_based_event_time_start(tmp_path):
    timestamps = _hourly("2024-01-01T00:00:00", 6)
    index_path = _write_index(tmp_path, timestamps)
    # Flare starts inside a short window measured from sequence_end but
    # peaks just after it -- only event_time="start" should catch it.
    catalog_path = _catalog_with_flare(tmp_path, "2024-01-01", "0505", "0510", "0515", "M1.0", "4456")

    builder = DatasetBuilder(
        prediction_window=0.1,  # 6 minutes: [05:00, 05:06) catches start=05:05, not peak=05:10
        strategy=BinaryThresholdStrategy(),
        cadence="1h", observation_window="6h", sliding_window="6h",
        event_time="start",
    )
    result = builder.build(index_path, catalog_path)
    assert result.iloc[0]["label"] == 1


def test_time_based_event_time_peak(tmp_path):
    timestamps = _hourly("2024-01-01T00:00:00", 6)
    index_path = _write_index(tmp_path, timestamps)
    catalog_path = _catalog_with_flare(tmp_path, "2024-01-01", "0459", "0501", "0502", "M1.0", "4456")

    builder = DatasetBuilder(
        prediction_window=24, strategy=BinaryThresholdStrategy(),
        cadence="1h", observation_window="6h", sliding_window="6h",
        event_time="peak",
    )
    result = builder.build(index_path, catalog_path)
    assert result.iloc[0]["label"] == 1


def test_time_based_window_excludes_its_end(tmp_path):
    timestamps = _hourly("2024-01-01T00:00:00", 6)
    index_path = _write_index(tmp_path, timestamps)
    # Flare peaks exactly at sequence_end + prediction_window -- excluded by [t, t+w).
    catalog_path = _catalog_with_flare(tmp_path, "2024-01-01", "0600", "0600", "0605", "X1.0", "4456")

    builder = DatasetBuilder(
        prediction_window=1, strategy=BinaryThresholdStrategy(),
        cadence="1h", observation_window="6h", sliding_window="6h",
    )
    result = builder.build(index_path, catalog_path)
    assert result.iloc[0]["label"] == 0


def test_time_based_max_flare_strategy(tmp_path):
    timestamps = _hourly("2024-01-01T00:00:00", 6)
    index_path = _write_index(tmp_path, timestamps)
    catalog_path = _catalog_with_flare(tmp_path, "2024-01-01", "0501", "0501", "0505", "M2.3", "4456")

    builder = DatasetBuilder(
        prediction_window=24, strategy=MaxFlareStrategy(),
        cadence="1h", observation_window="6h", sliding_window="6h",
    )
    result = builder.build(index_path, catalog_path)
    assert result.iloc[0]["label"] == 2.3e-5


def test_time_based_preserves_extra_columns(tmp_path):
    timestamps = _hourly("2024-01-01T00:00:00", 6)
    index_path = _write_index(tmp_path, timestamps, extra_columns={
        "image_path": [f"img_{i}.jpg" for i in range(6)],
    })
    catalog_path = _empty_catalog(tmp_path)

    builder = DatasetBuilder(
        prediction_window=24, strategy=BinaryThresholdStrategy(),
        cadence="1h", observation_window="6h", sliding_window="6h",
    )
    result = builder.build(index_path, catalog_path)

    assert "image_path" in result.columns
    assert result.iloc[0]["image_path"] == [f"img_{i}.jpg" for i in range(6)]


# ── Active-region time-based mode ─────────────────────────────────────────


def test_active_region_time_based_basic(tmp_path):
    timestamps = _hourly("2024-01-01T08:00:00", 6)
    index_path = _write_index(tmp_path, timestamps, active_regions=["3559"] * 6)
    catalog_path = _catalog_with_flare(tmp_path, "2024-01-01", "1301", "1301", "1305", "M1.0", "3559")

    builder = DatasetBuilder(
        prediction_window=1, strategy=BinaryThresholdStrategy(), target="active_region",
        cadence="1h", observation_window="6h", sliding_window="6h",
    )
    result = builder.build(index_path, catalog_path)

    assert list(result.columns) == [
        "sequence_start", "sequence_end", "timestamps", "n_images", "active_region", "label",
    ]
    assert len(result) == 1
    assert result.iloc[0]["active_region"] == "3559"
    assert result.iloc[0]["sequence_end"] == pd.Timestamp("2024-01-01T13:00:00")
    assert result.iloc[0]["label"] == 1


def test_active_region_time_based_regions_are_grouped(tmp_path):
    # Grouped by region: region 3559 lacks 02:00 (that image is 9999's),
    # so its only 6-hour candidate is skipped; 9999 has a single image.
    timestamps = _hourly("2024-01-01T00:00:00", 6)
    index_path = _write_index(
        tmp_path, timestamps, active_regions=["3559", "3559", "9999", "3559", "3559", "3559"]
    )
    catalog_path = _empty_catalog(tmp_path)

    builder = DatasetBuilder(
        prediction_window=24, strategy=BinaryThresholdStrategy(), target="active_region",
        cadence="1h", observation_window="6h", sliding_window="6h",
    )
    result, report = builder.build(index_path, catalog_path, return_report=True)
    assert result.empty
    assert report.region_count == 2
    assert report.skipped_sequences == 1
    assert report.skipped_sequence_regions == ["3559"]
    assert report.missing_timestamps == [pd.Timestamp("2024-01-01T02:00:00")]


def test_active_region_time_based_missing_active_region_breaks_the_region_grid(tmp_path):
    timestamps = _hourly("2024-01-01T00:00:00", 6)
    index_path = _write_index(
        tmp_path, timestamps, active_regions=["3559", "3559", None, "3559", "3559", "3559"]
    )
    catalog_path = _catalog_with_flare(tmp_path, "2024-01-01", "0500", "0500", "0505", "X1.0", "3559")

    builder = DatasetBuilder(
        prediction_window=1, strategy=BinaryThresholdStrategy(), target="active_region",
        cadence="1h", observation_window="6h", sliding_window="6h",
    )
    with pytest.warns(UserWarning, match="missing active_region"):
        result = builder.build(index_path, catalog_path)

    # The 02:00 image has no region, so region 3559's grid is incomplete
    # and its sequence is skipped -- never labeled from the X1.0 flare.
    assert result.empty


def test_active_region_time_based_with_report(tmp_path):
    timestamps = _hourly("2024-01-01T00:00:00", 6)
    index_path = _write_index(tmp_path, timestamps, active_regions=["3559"] * 6)
    catalog_path = _empty_catalog(tmp_path)

    builder = DatasetBuilder(
        prediction_window=24, strategy=BinaryThresholdStrategy(), target="active_region",
        cadence="1h", observation_window="6h", sliding_window="6h",
    )
    result, report = builder.build(index_path, catalog_path, return_report=True)

    assert isinstance(report, SequenceBuildReport)
    assert report.emitted_sequences == 1
    assert "image_path" not in result.columns  # no extra columns given
