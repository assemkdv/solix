import solix


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

    assert set(solix.__all__) == expected
    for name in expected:
        assert hasattr(solix, name)


def test_engine_is_not_public():
    assert not hasattr(solix, "DatasetBuilder")


def test_top_level_classes_are_usable():
    builder = solix.FullDiskSingleInstance(prediction_window=24, reducer=solix.MaxFluxReducer(), labeler=solix.BinaryThresholdLabeler("M"))
    assert builder.prediction_window == 24
