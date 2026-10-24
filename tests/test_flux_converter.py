import math
import pytest
from flare_indexer.goes_class import FluxConverter

# ── FluxConverter tests ────────────────────────────────────────────────


def test_to_flux():
    converter = FluxConverter()
    assert converter.to_flux("M2.3") == 2.3e-5
    assert converter.to_flux("C4.0") == 4e-6
    assert converter.to_flux("X1.0") == 1e-4
    assert converter.to_flux("A1.0") == 1e-8


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
    from flare_indexer import BinaryThresholdLabeler

    converter = FluxConverter()
    labeler = BinaryThresholdLabeler("M")
    assert converter.to_flux("C10.0") == 1e-05
    assert labeler.assign(converter.to_flux("C10.0")) == 1
    # The threshold comparison itself stays exact, with no tolerance.
    assert labeler.assign(math.nextafter(1e-05, 0.0)) == 0
    assert labeler.assign(math.nextafter(1e-05, math.inf)) == 1


# ── FluxConverter.to_class (flux -> letter) ─────────────────────────────


def test_to_class_boundaries_belong_to_the_higher_class():
    converter = FluxConverter()
    for letter in FluxConverter.CLASS_LETTERS:
        boundary = FluxConverter.FLUX_MAP[letter]
        assert converter.to_class(boundary) == letter
        assert converter.to_class(boundary * 2) == letter
        below = math.nextafter(boundary, 0)
        if letter == "A":
            with pytest.raises(ValueError, match="below the A-class boundary"):
                converter.to_class(below)
        else:
            lower = FluxConverter.CLASS_LETTERS[FluxConverter.CLASS_LETTERS.index(letter) - 1]
            assert converter.to_class(below) == lower
    assert converter.to_class(1.0) == "X"


@pytest.mark.parametrize("text, letter", [("C10.0", "M"), ("M1.0", "M"), ("M10.0", "X"), ("X1.0", "X"),
                                          ("M9.9", "M"), ("A1.0", "A"), ("B99", "C")])
def test_to_class_round_trips_equivalent_class_text(text, letter):
    converter = FluxConverter()
    assert converter.to_class(converter.to_flux(text)) == letter


@pytest.mark.parametrize("value", [0, 0.0, -1e-6, float("nan"), float("inf")])
def test_to_class_rejects_values_that_are_not_a_flare_flux(value):
    with pytest.raises(ValueError):
        FluxConverter().to_class(value)


@pytest.mark.parametrize("value", ["M2.0", None, True, [1e-5]])
def test_to_class_rejects_non_numbers(value):
    with pytest.raises(TypeError, match="expects a flux number"):
        FluxConverter().to_class(value)


def test_class_letters_match_flux_map_in_increasing_order():
    letters = FluxConverter.CLASS_LETTERS
    assert letters == tuple(FluxConverter.FLUX_MAP) == ("A", "B", "C", "M", "X")
    fluxes = [FluxConverter.FLUX_MAP[l] for l in letters]
    assert fluxes == sorted(fluxes)
