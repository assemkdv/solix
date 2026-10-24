import math
import numbers
import re
from decimal import Decimal

import pandas as pd

# A GOES class is one letter A/B/C/M/X, optionally followed by a positive
# multiplier ("M", "M2", "M2.3"). Case and surrounding whitespace are
# tolerated; anything else is rejected rather than guessed at.
_GOES_CLASS_PATTERN = re.compile(r"^\s*([ABCMX])\s*(\d+(?:\.\d+)?)?\s*$", re.IGNORECASE)

# Cap on how many invalid catalog values an error message lists verbatim.
_MAX_INVALID_EXAMPLES = 5


class InvalidGOESClassError(ValueError):
    """Raised for a missing or malformed GOES class value."""


def parse_goes_class(goes_class) -> tuple[str, float]:
    """
    Split a GOES class into (letter, multiplier), e.g. "M2.3" -> ("M", 2.3).

    A missing multiplier means 1.0 ("M" -> ("M", 1.0)). Raises
    InvalidGOESClassError for a missing value (None/NaN/blank), an unknown
    letter, a malformed number ("M2.3.4"), or a multiplier of zero.
    """
    letter, multiplier = _parse_goes_class_decimal(goes_class)
    return letter, float(multiplier)


def _parse_goes_class_decimal(goes_class) -> tuple[str, Decimal]:
    """parse_goes_class, with the multiplier as the exact Decimal written in the text."""
    if not isinstance(goes_class, str):
        if goes_class is None or pd.isna(goes_class):
            raise InvalidGOESClassError("GOES class is missing (empty or NaN); expected e.g. 'M2.3'.")
        raise InvalidGOESClassError(
            f"GOES class must be a string such as 'M2.3', got {goes_class!r} ({type(goes_class).__name__})."
        )
    match = _GOES_CLASS_PATTERN.match(goes_class)
    if match is None:
        if not goes_class.strip():
            raise InvalidGOESClassError("GOES class is missing (empty string); expected e.g. 'M2.3'.")
        raise InvalidGOESClassError(
            f"Invalid GOES class {goes_class!r}: expected a letter A, B, C, M or X followed by an "
            "optional positive number, e.g. 'M2.3'."
        )
    letter = match.group(1).upper()
    multiplier = Decimal(match.group(2)) if match.group(2) is not None else Decimal(1)
    if multiplier <= 0:
        raise InvalidGOESClassError(f"Invalid GOES class {goes_class!r}: the number must be positive.")
    return letter, multiplier


def validate_goes_class_column(values: pd.Series, source, column: str = "class") -> None:
    """
    Check every value of a catalog's GOES class column, raising one
    InvalidGOESClassError that counts all bad rows and lists the first few
    (row number in the file, counting the header as line 1, and value).
    """
    invalid = []
    for position, value in enumerate(values):
        try:
            parse_goes_class(value)
        except InvalidGOESClassError:
            invalid.append((position, value))
    if not invalid:
        return
    examples = ", ".join(
        f"line {position + 2}: {'<missing>' if not isinstance(value, str) and pd.isna(value) else repr(value)}"
        for position, value in invalid[:_MAX_INVALID_EXAMPLES]
    )
    more = ", ..." if len(invalid) > _MAX_INVALID_EXAMPLES else ""
    raise InvalidGOESClassError(
        f"Flare catalog at {source} has {len(invalid)} missing or invalid GOES class value(s) in "
        f"column '{column}' ({examples}{more}). Expected a letter A, B, C, M or X followed by an "
        "optional positive number, e.g. 'M2.3'. Fix or remove these rows explicitly: a flare of "
        "unknown class cannot be compared with a threshold."
    )


class FluxConverter:
    """
    Converts GOES flare class letters to numeric X-ray flux values (W/m²).

    Scale (logarithmic):
        A → 1e-8
        B → 1e-7
        C → 1e-6
        M → 1e-5
        X → 1e-4

    to_flux() multiplies in decimal arithmetic, from the class text, and
    rounds to a float only once at the end. Equal values written
    differently therefore give the identical float: "C10.0" and "M1.0" are
    both exactly 1e-05, "M10.0" and "X1.0" both 1e-04. (Multiplying the
    floats instead, 1e-6 * 10.0 = 9.999999999999999e-06, is just below
    1e-05 and would fall below an M1.0 threshold.)
    """

    CLASS_LETTERS = ("A", "B", "C", "M", "X")   # weakest to strongest

    FLUX_MAP = {
        "A": 1e-8,
        "B": 1e-7,
        "C": 1e-6,
        "M": 1e-5,
        "X": 1e-4,
    }

    def to_flux(self, goes_class: str) -> float:
        """
        Convert a GOES class string to numeric flux.

        Examples: "M2.3" → 2.3e-5,  "C4" → 4e-6,  "m2.3" → 2.3e-5,
        "C10.0" → 1e-5 (identical to "M1.0").

        The result is the float nearest to the exact decimal product
        multiplier × FLUX_MAP[letter] (FLUX_MAP values taken as the
        decimals they are written as, e.g. 1e-05).

        Raises InvalidGOESClassError (a ValueError) for a missing or
        malformed class -- see parse_goes_class().
        """
        letter, multiplier = _parse_goes_class_decimal(goes_class)
        return float(multiplier * Decimal(repr(self.FLUX_MAP[letter])))

    def to_class(self, flux) -> str:
        """
        The GOES class letter of a flux value: the opposite direction of
        to_flux(), decided by the lower boundary of each class in FLUX_MAP.

        A >= 1e-8, B >= 1e-7, C >= 1e-6, M >= 1e-5, X >= 1e-4 (no upper
        bound); each boundary belongs to the higher class, and the
        comparison is exact (one float step below 1e-5 is "C").
        Examples: 2e-5 → "M", 1e-5 → "M", 9.999999999999999e-06 → "C".

        Raises ValueError for a positive flux below the A boundary (the
        scale has no letter for it), for zero (no flare, not a class), and
        for a negative, NaN or infinite value; TypeError for a non-number
        (including text and booleans).
        """
        if isinstance(flux, bool) or not isinstance(flux, numbers.Real):
            raise TypeError(f"FluxConverter.to_class expects a flux number in W/m^2, got {flux!r} "
                            f"({type(flux).__name__}).")
        value = float(flux)
        if not math.isfinite(value):
            raise ValueError(f"FluxConverter.to_class got a non-finite flux {flux!r}.")
        if value <= 0:
            raise ValueError(f"FluxConverter.to_class got {value!r}; a GOES class needs a positive flux "
                             "(0.0 means no flare, and X-ray flux cannot be negative).")
        for letter in reversed(self.CLASS_LETTERS):
            if value >= self.FLUX_MAP[letter]:
                return letter
        raise ValueError(f"FluxConverter.to_class got a positive flux {value!r} W/m^2 below the A-class "
                         f"boundary {self.FLUX_MAP['A']!r} W/m^2; the GOES scale has no class for it.")

