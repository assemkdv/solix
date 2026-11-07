import pandas as pd
import pytest

import flare_indexer as f
from flare_indexer.builder import DatasetBuilder


def events(classes):
    return [f.FlareEvent(pd.Timestamp('2024-01-01 01:05'), c,
                        pd.Timestamp('2024-01-01 01:00'), '100') for c in classes]


def test_minimum_numeric_and_empty():
    assert f.MinFluxReducer().reduce(events(['M10', 'M9', 'C10'])) == 1e-5
    assert f.MinFluxReducer().reduce([]) == 0.0
    assert f.MinFluxReducer().reduce(events(['C4'])) == 4e-6


def test_mean_of_event_fluxes_and_empty():
    assert f.MeanFluxReducer().reduce(events(['C1', 'C3', 'M2'])) == pytest.approx((1e-6 + 3e-6 + 2e-5) / 3)
    assert f.MeanFluxReducer().reduce(events(['C10', 'M1'])) == pytest.approx(1e-5)
    assert f.MeanFluxReducer().reduce(events(['X2.9'])) == pytest.approx(2.9e-4)
    assert f.MeanFluxReducer().reduce([]) == 0.0


def test_mean_flux_in_builder(tmp_path):
    catalog = tmp_path / "catalog.csv"
    catalog.write_text(
        "start_time,peak_time,end_time,class,active_region\n"
        "2024-01-01 02:00:00,2024-01-01 02:10:00,2024-01-01 02:20:00,C1.0,100\n"
        "2024-01-01 05:00:00,2024-01-01 05:10:00,2024-01-01 05:20:00,M2.0,100\n"
    )
    index = tmp_path / "index.csv"
    index.write_text("timestamp\n2024-01-01 00:00:00\n2024-01-01 03:00:00\n2024-01-02 06:00:00\n")
    result = DatasetBuilder(
        "24h", reducer=f.MeanFluxReducer(), labeler=f.RegressionLabeler()
    ).build(index, catalog)
    assert result["label"].tolist() == pytest.approx([(1e-6 + 2e-5) / 2, 2e-5, 0.0])


def test_counts_retain_every_event_and_normalize_class():
    assert f.ClassCountReducer().reduce(events(['C2'] * 10 + ['X1', 'C10', 'M10'])) == {
        'A': 0, 'B': 0, 'C': 10, 'M': 1, 'X': 2}
    assert f.ClassCountReducer().reduce([]) == dict.fromkeys('ABCMX', 0)
    with pytest.raises(ValueError):
        f.ClassCountReducer().reduce(events(['A0.5']))


def test_labelers_copy_and_presence():
    counts = dict(zip('ABCMX', [0, 2, 10, 0, 1]))
    result = f.ClassCountLabeler().assign(counts)
    result['B'] = 9
    assert counts['B'] == 2
    assert f.ClassPresenceLabeler().assign(counts) == dict(zip('ABCMX', [0, 1, 1, 0, 1]))
    assert f.ClassPresenceLabeler().assign(dict.fromkeys('ABCMX', 0)) == dict.fromkeys('ABCMX', 0)


@pytest.mark.parametrize('value,error', [
    (1e-5, TypeError), ({'M': 1}, ValueError),
    (dict.fromkeys('ABCMX', -1), ValueError),
    (dict.fromkeys('ABCMX', True), TypeError),
    (dict.fromkeys('ABCMX', 1.5), TypeError),
])
@pytest.mark.parametrize('labeler', [f.ClassCountLabeler, f.ClassPresenceLabeler])
def test_invalid_counts(value, error, labeler):
    with pytest.raises(error):
        labeler().assign(value)


@pytest.mark.parametrize('target', ['full_disk', 'active_region'])
@pytest.mark.parametrize('mode', ['single', 'time'])
@pytest.mark.parametrize('labeler', [f.ClassCountLabeler, f.ClassPresenceLabeler])
def test_builder_all_modes(tmp_path, target, mode, labeler):
    index = tmp_path / 'instances.csv'
    catalog = tmp_path / 'catalog.csv'
    pd.DataFrame({'timestamp': ['2024-01-01 00:00', '2024-01-01 01:00'],
                  'active_region': [100, 100]}).to_csv(index, index=False)
    pd.DataFrame({'start_time': ['2024-01-01 00:30', '2024-01-01 01:30', '2024-01-01 01:30'],
                  'peak_time': ['2024-01-01 00:35', '2024-01-01 01:35', '2024-01-01 01:35'],
                  'end_time': ['2024-01-01 00:40', '2024-01-01 01:40', '2024-01-01 01:40'],
                  'class': ['B2', 'C2', 'X1'], 'active_region': [100, 100, 200]}).to_csv(catalog, index=False)
    options = {} if mode == 'single' else {'cadence': '1h', 'observation_window': '2h', 'sliding_window': '1h'}
    result = DatasetBuilder(
        '1h', reducer=f.ClassCountReducer(), labeler=labeler(), target=target, **options,
    ).build(index, catalog)
    assert len(result) == (2 if mode == 'single' else 1)
    assert result.iloc[-1].label == {'A': 0, 'B': 0, 'C': 1, 'M': 0, 'X': int(target == 'full_disk')}
    if mode == 'single':
        assert result.iloc[0].label == {'A': 0, 'B': 1, 'C': 0, 'M': 0, 'X': 0}


def test_minimum_pipeline_builder(tmp_path):
    index, catalog = tmp_path / 'instances.csv', tmp_path / 'catalog.csv'
    pd.DataFrame({'timestamp': ['2024-01-01']}).to_csv(index, index=False)
    pd.DataFrame({'start_time': ['2024-01-01 01:00'] * 2,
                  'peak_time': ['2024-01-01 01:05'] * 2, 'end_time': ['2024-01-01 01:10'] * 2,
                  'class': ['C2', 'X1'], 'active_region': [100, 100]}).to_csv(catalog, index=False)
    builder = DatasetBuilder(24, reducer=f.MinFluxReducer(), labeler=f.RegressionLabeler())
    assert builder.build(index, catalog).iloc[0].label == 2e-6
