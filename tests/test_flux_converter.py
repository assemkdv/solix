import pytest
from flare_indexer.classifier import FluxConverter

# ── FluxConverter tests ────────────────────────────────────────────────


def test_to_flux():
    converter = FluxConverter()
    assert converter.to_flux("M2.3") == 2.3e-5
    assert converter.to_flux("C4.0") == 4e-6
    assert converter.to_flux("X1.0") == 1e-4
    assert converter.to_flux("A1.0") == 1e-8


def test_to_flux_invalid_letter_raises():
    converter = FluxConverter()
    with pytest.raises(KeyError):
        converter.to_flux("Z5.0")


def test_to_flux_empty_string_raises():
    converter = FluxConverter()
    with pytest.raises(IndexError):
        converter.to_flux("")


def test_to_flux_non_numeric_suffix_raises():
    converter = FluxConverter()
    with pytest.raises(ValueError):
        converter.to_flux("M2.3.4")


def test_to_flux_numeric_ordering_not_lexical():
    # A lexical ("string") comparison of "M9.0" vs "M10.0" would incorrectly
    # rank "M9.0" higher ('9' > '1' as characters) -- the legacy lab script
    # has exactly this bug. FluxConverter must order them numerically:
    # M10.0 (10x the M-class base) is stronger than M9.0 (9x), and in fact
    # exactly as strong as X1.0.
    converter = FluxConverter()
    assert converter.to_flux("M9.0") < converter.to_flux("M10.0")
    assert converter.to_flux("M10.0") == converter.to_flux("X1.0")
    assert sorted(["M10.0", "M9.0", "X1.0"], key=converter.to_flux) == ["M9.0", "M10.0", "X1.0"]


@pytest.mark.parametrize("text, equivalent", [
    ("A10.0", "B1.0"), ("B10.0", "C1.0"), ("C10.0", "M1.0"), ("M10.0", "X1.0"), ("C100", "X1"), ("m2.30", "M2.3"),
])
def test_equivalent_class_text_gives_identical_flux(text, equivalent):
    # Decimal arithmetic from the class text: identical floats, not merely close
    # ones (multiplying floats gives 1e-6 * 10.0 = 9.999999999999999e-06).
    converter = FluxConverter()
    assert converter.to_flux(text) == converter.to_flux(equivalent)


def test_binary_m_threshold_at_c10_and_adjacent_floats():
    import math

    from flare_indexer import BinaryThresholdLabeler, FlareClassifier

    converter = FluxConverter()
    labeler = BinaryThresholdLabeler("M")
    assert converter.to_flux("C10.0") == 1e-05
    assert labeler.assign(converter.to_flux("C10.0")) == 1
    assert FlareClassifier().is_strong("C10.0", "M")
    # The threshold comparison itself stays exact, with no tolerance.
    assert labeler.assign(math.nextafter(1e-05, 0.0)) == 0
    assert labeler.assign(math.nextafter(1e-05, math.inf)) == 1
