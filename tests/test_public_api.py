import flare_indexer as fidx


def test_public_names_are_accessible():
    expected = {
        "FluxConverter",
        "InvalidGOESClassError",
        "FlareEvent",
        "EventMatcher",
        "EventReducer",
        "MaxFluxReducer",
        "MinFluxReducer",
        "MeanFluxReducer",
        "ClassCountReducer",
        "CumulativeClassIndexReducer",
        "MaxRelativeFluxReducer",
        "SumRelativeFluxReducer",
        "ClassCountLabeler",
        "ClassPresenceLabeler",
        "LabelAssigner",
        "BinaryThresholdLabeler",
        "NumericThresholdLabeler",
        "RegressionLabeler",
        "MulticlassLabeler",
        "PipelineResult",
        "DatasetBuildingPipeline",
        "FullDiskSingleInstance",
        "FullDiskSequence",
        "ActiveRegionSingleInstance",
        "ActiveRegionSequence",
        "SequenceBuildReport",
        "build_index_from_hmi_filenames",
        "adapt_goes_catalog",
        "add_relative_flux",
        "HMIIndexReport",
    }

    assert set(fidx.__all__) == expected
    for name in expected:
        assert hasattr(fidx, name)


def test_engine_is_not_public():
    assert not hasattr(fidx, "DatasetBuilder")


def test_top_level_classes_are_usable():
    builder = fidx.FullDiskSingleInstance(prediction_window=24, reducer=fidx.MaxFluxReducer(), labeler=fidx.BinaryThresholdLabeler("M"))
    assert builder.prediction_window == 24
