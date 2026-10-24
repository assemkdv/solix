import flare_indexer as fidx


def test_public_names_are_accessible():
    expected = {
        "ThresholdChecker",
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
        "BinaryThresholdStrategy",
        "MaxFlareStrategy",
        "PipelineResult",
        "DatasetBuildingPipeline",
        "FullDiskSingleImage",
        "FullDiskSequence",
        "ActiveRegionSingleImage",
        "ActiveRegionSequence",
        "SequenceBuildReport",
        "build_image_index_from_filenames",
        "adapt_goes_catalog",
        "add_relative_flux",
        "ImageIndexReport",
    }

    assert set(fidx.__all__) == expected
    for name in expected:
        assert hasattr(fidx, name)


def test_engine_is_not_public():
    assert not hasattr(fidx, "DatasetBuilder")


def test_top_level_classes_are_usable():
    strategy = fidx.BinaryThresholdStrategy()
    assert strategy.threshold == "M"

    builder = fidx.FullDiskSingleImage(prediction_window=24, strategy=fidx.BinaryThresholdStrategy())
    assert builder.prediction_window == 24
