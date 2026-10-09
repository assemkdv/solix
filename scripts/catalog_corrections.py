"""
Documented corrections to the lab's integrated GOES catalog, applied by the
validation scripts before labeling. The upstream file is never modified: the
corrected copy is written to scripts/.cache/.

Problem. goes_flares_integrated.csv (explainingFullDisk and arpatch-sfp, SHA-256
2a1c76ff...cf48) has seven flares that start a few minutes before midnight and
end after it, whose peak_time is 00:00 on the START date, i.e. before the start.
The same values are in the HEK SWPC flare list the catalog was built from, so
the error is upstream of the lab, not in flare_indexer. NOAA's own GOES X-ray
reports (NGDC, bare HHMM times) record the maximum as 0000 for six of them --
the only midnight inside the flare is the start of the next day -- and as ////
(missing) for the 2011-04-11 C2.5 flare.

Corrections. The six: peak_time -> 00:00 on the next day. The 2011-04-11 flare:
peak_time -> missing (a flare without a peak is still matched by start time, as
the lab's labeling and both validation scripts do; it can never be matched by
peak time). Every record is matched on all of its original values and the
match must be unique, so a changed upstream file fails loudly instead of being
silently "corrected". The published labels were made by start time and are not
affected by these peaks.

Source: https://www.ngdc.noaa.gov/stp/space-weather/solar-data/solar-features/
solar-flares/x-rays/goes/xrs/goes-xrs-report_<year>.txt (2017: _2017-ytd.txt),
columns "yymmdd start end max". The exact report lines are quoted below.
"""

import hashlib
from dataclasses import dataclass, asdict
from pathlib import Path

import pandas as pd

NGDC_URL = ("https://www.ngdc.noaa.gov/stp/space-weather/solar-data/solar-features/"
            "solar-flares/x-rays/goes/xrs/")
EXPECTED_SHA256 = "2a1c76ff6ba508fede2add75bc7d4a96a61283d676cc360ae13d5aa1d56dcf48"
MISSING = None  # corrected_peak_time value meaning "no peak time"


@dataclass(frozen=True)
class Correction:
    start_time: str              # original values; all must match exactly
    peak_time: str
    end_time: str
    goes_class: str
    noaa_active_region: int
    corrected_peak_time: str | None
    ngdc_file: str
    ngdc_line: str               # "yymmdd start end max" as in the NOAA report
    reason: str


_NEXT_DAY_MIDNIGHT = "NOAA reports the maximum at 0000; the only midnight within the flare is the next day's."

CORRECTIONS = (
    Correction("2011-04-11 23:40:00", "2011-04-11 00:00:00", "2011-04-12 00:03:00", "C2.5", 11190, MISSING,
               "goes-xrs-report_2011.txt", "110411 2340 0003 ////",
               "NOAA records no maximum time (////); the catalog's 00:00 is a fill value, so the peak is unknown."),
    Correction("2012-04-05 23:50:00", "2012-04-05 00:00:00", "2012-04-06 00:05:00", "C1.1", 11450,
               "2012-04-06 00:00:00", "goes-xrs-report_2012.txt", "120405 2350 0005 0000", _NEXT_DAY_MIDNIGHT),
    Correction("2012-06-06 23:56:00", "2012-06-06 00:00:00", "2012-06-07 00:05:00", "C1.0", 11494,
               "2012-06-07 00:00:00", "goes-xrs-report_2012.txt", "120606 2356 0005 0000", _NEXT_DAY_MIDNIGHT),
    Correction("2012-07-30 23:50:00", "2012-07-30 00:00:00", "2012-07-31 00:04:00", "C6.3", 11538,
               "2012-07-31 00:00:00", "goes-xrs-report_2012.txt", "120730 2350 0004 0000", _NEXT_DAY_MIDNIGHT),
    Correction("2012-10-13 23:46:00", "2012-10-13 00:00:00", "2012-10-14 00:18:00", "C1.5", 11589,
               "2012-10-14 00:00:00", "goes-xrs-report_2012.txt", "121013 2346 0018 0000", _NEXT_DAY_MIDNIGHT),
    Correction("2014-05-25 23:49:00", "2014-05-25 00:00:00", "2014-05-26 00:04:00", "C3.8", 12065,
               "2014-05-26 00:00:00", "goes-xrs-report_2014.txt", "140525 2349 0004 0000", _NEXT_DAY_MIDNIGHT),
    Correction("2017-03-26 23:49:00", "2017-03-26 00:00:00", "2017-03-27 00:08:00", "C1.3", 12644,
               "2017-03-27 00:00:00", "goes-xrs-report_2017-ytd.txt", "170326 2349 0008 0000", _NEXT_DAY_MIDNIGHT),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _matches(df: pd.DataFrame, c: Correction) -> pd.Series:
    text = {col: df[col].astype(str).str.strip() for col in ("start_time", "peak_time", "end_time", "goes_class")}
    region = pd.to_numeric(df["noaa_active_region"], errors="coerce")
    return ((text["start_time"] == c.start_time) & (text["peak_time"] == c.peak_time)
            & (text["end_time"] == c.end_time) & (text["goes_class"] == c.goes_class)
            & (region == c.noaa_active_region))


def apply_catalog_corrections(source_path, output_path) -> tuple[Path, dict]:
    """
    Write a corrected copy of goes_flares_integrated.csv and describe every change.

    Each correction's original values (start, peak, end, class, NOAA region)
    must match exactly one row, otherwise ValueError: nothing is written.
    Returns (output_path, report); report["corrections"] has one entry per
    applied correction with the data row (1 = first row after the header), the
    original and corrected peak and the NOAA source line.
    """
    source_path, output_path = Path(source_path), Path(output_path)
    df = pd.read_csv(source_path)
    required = {"start_time", "peak_time", "end_time", "goes_class", "noaa_active_region"}
    if missing := required - set(df.columns):
        raise ValueError(f"{source_path} is missing column(s): {', '.join(sorted(missing))}")

    peak = df["peak_time"].astype(object)
    applied = []
    for c in CORRECTIONS:
        hits = _matches(df, c)
        if hits.sum() != 1:
            raise ValueError(
                f"{source_path}: expected exactly one row with start_time={c.start_time}, "
                f"peak_time={c.peak_time}, end_time={c.end_time}, goes_class={c.goes_class}, "
                f"noaa_active_region={c.noaa_active_region}, found {int(hits.sum())}. The catalog differs "
                f"from the one these corrections were written for (SHA-256 {EXPECTED_SHA256[:12]}...)."
            )
        position = int(hits.to_numpy().nonzero()[0][0])
        peak.iloc[position] = "" if c.corrected_peak_time is MISSING else c.corrected_peak_time
        applied.append({"data_row": position + 1, "goes_class": c.goes_class,
                        "noaa_active_region": c.noaa_active_region, "start_time": c.start_time,
                        "original_peak_time": c.peak_time, "corrected_peak_time": c.corrected_peak_time,
                        "end_time": c.end_time, "ngdc_file": c.ngdc_file, "ngdc_line": c.ngdc_line,
                        "reason": c.reason})
    df["peak_time"] = peak

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    sha = _sha256(source_path)
    report = {
        "source": str(source_path),
        "source_sha256": sha,
        "source_is_the_known_catalog": sha == EXPECTED_SHA256,
        "corrected_copy": str(output_path),
        "ngdc_url": NGDC_URL,
        "rule": "peak_time -> next-day 00:00 where NOAA max is 0000; -> missing where NOAA max is ////",
        "corrections": applied,
    }
    return output_path, report


def print_corrections(report: dict) -> None:
    print(f"  source: {report['source']} (SHA-256 {report['source_sha256'][:12]}..., "
          f"{'the known catalog' if report['source_is_the_known_catalog'] else 'NOT the known catalog'})")
    print(f"  corrected copy: {report['corrected_copy']}")
    print(f"  {len(report['corrections'])} peak_time correction(s), NOAA source {report['ngdc_url']}")
    for c in report["corrections"]:
        new = c["corrected_peak_time"] or "missing"
        print(f"    row {c['data_row']:>5} {c['goes_class']} AR {c['noaa_active_region']}: start {c['start_time']}, "
              f"end {c['end_time']}; peak {c['original_peak_time']} -> {new}   [{c['ngdc_file']}: {c['ngdc_line']}]")


def corrections_table() -> pd.DataFrame:
    return pd.DataFrame([asdict(c) for c in CORRECTIONS])
