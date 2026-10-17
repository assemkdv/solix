"""
Command-line interface: ``flare-indexer label IMAGES.csv CATALOG.csv --output OUT.csv``.

A thin wrapper around DatasetBuilder and the built-in strategies and
pipelines. Everything scientific (window matching, reduction, labeling)
happens in the Python API; this module only turns arguments into those
objects, writes the result as CSV (list and dict cells as JSON text) and
reports warnings and errors on stderr.
"""

import argparse
import dataclasses
import json
import os
import sys
import tempfile
import warnings
from importlib.metadata import PackageNotFoundError, version as _distribution_version
from pathlib import Path

import numpy as np
import pandas as pd

from .builder import DatasetBuilder
from .labelers import (
    ClassCountLabeler, ClassPresenceLabeler, MulticlassLabeler, NumericThresholdLabeler, RegressionLabeler,
)
from .pipeline import DatasetBuildingPipeline
from .reducers import ClassCountReducer, CumulativeClassIndexReducer, MaxFluxReducer
from .strategies import BinaryThresholdStrategy, MaxFlareStrategy

LABELINGS = ("binary", "max-flux", "multiclass", "class-counts", "class-presence", "cumulative")

# Python parameter names as they appear in DatasetBuilder's messages -> CLI flags.
_FLAG_NAMES = {
    "prediction_window": "--window",
    "sequence_length": "--sequence-length",
    "stride": "--stride",
    "cadence_minutes": "--cadence-minutes",
    "start": "--start",
    "cadence": "--cadence",
    "observation_window": "--observation-window",
    "sliding_window": "--sliding-window",
    "missing_active_region": "--missing-active-region",
    "event_time": "--event-time",
    "interval_mode": "--interval-mode",
}
_ROW_BASED = ("sequence_length", "stride", "cadence_minutes")
_TIME_BASED = ("start", "cadence", "observation_window", "sliding_window")
_JSON_CELL_TYPES = (list, tuple, dict, np.ndarray)


class CLIError(Exception):
    """A problem the command line can explain in one line (exit status 1)."""


def _version_text() -> str:
    try:
        return _distribution_version("flare-indexer")
    except PackageNotFoundError:
        return "unknown (package not installed)"


def _positive_int(text: str) -> int:
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected a whole number, got {text!r}") from None
    if value < 1:
        raise argparse.ArgumentTypeError(f"expected a number >= 1, got {text!r}")
    return value


def _positive_number(text: str):
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected a number, got {text!r}") from None
    if not value > 0:
        raise argparse.ArgumentTypeError(f"expected a number > 0, got {text!r}")
    return int(value) if value.is_integer() else value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="flare-indexer",
        description="Label solar image indexes with GOES flare events (command-line front end of the "
                    "flare_indexer Python package).",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {_version_text()}")
    commands = parser.add_subparsers(dest="command", metavar="COMMAND", required=True)

    label = commands.add_parser(
        "label",
        help="label an image index CSV from a flare catalog CSV",
        description=(
            "Read IMAGES.csv (column timestamp [+ active_region] [+ other columns]) and CATALOG.csv "
            "(start_time, peak_time, end_time, class, active_region), label every image or sequence "
            "with DatasetBuilder, and write OUT.csv. Other index columns (image paths, ids) are kept. "
            "Cells holding a list or a dict (sequence timestamps, carried columns, class counts) are "
            "written as JSON text. Durations need a unit: 24h, 90min, 23h59min59s."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    label.add_argument("images", metavar="IMAGES.csv", help="image index CSV")
    label.add_argument("catalog", metavar="CATALOG.csv", help="flare catalog CSV")
    label.add_argument("--output", "-o", required=True, metavar="OUT.csv", help="where to write the labels")
    label.add_argument("--overwrite", action="store_true",
                       help="replace OUT.csv (and REPORT.json) if they already exist")
    label.add_argument("--labeling", choices=LABELINGS, default="binary",
                       help="binary: 0/1 by GOES threshold on the strongest flare; max-flux: peak flux in W/m^2; "
                            "multiclass: NF/A/B/C/M/X of the strongest flare; class-counts / class-presence: "
                            "per-class counts / 0-1 flags as a JSON dict; cumulative: cumulative class index "
                            "(sum C + 10 sum M + 100 sum X)")
    label.add_argument("--threshold", default=argparse.SUPPRESS, metavar="VALUE",
                       help="binary: GOES class threshold, e.g. M or C4.2 (default M); cumulative: a number, "
                            "turns the index into 0/1 (index >= VALUE; default: write the raw index); "
                            "not accepted with other labelings")
    label.add_argument("--window", default="24h", metavar="DURATION",
                       help="prediction window after each image (after the last image of a sequence)")
    label.add_argument("--event-time", choices=("peak", "start"), default="peak",
                       help="which flare time must fall in the window")
    label.add_argument("--interval-mode", choices=("left_closed", "right_closed"), default="left_closed",
                       help="left_closed = [t, t+w), right_closed = (t, t+w]")
    label.add_argument("--target", choices=("full_disk", "active_region"), default="full_disk",
                       help="active_region matches only flares of the image's active_region column")
    label.add_argument("--missing-active-region", choices=("negative", "drop", "error"), default=argparse.SUPPRESS,
                       help="active_region target only: what to do with images whose region is blank "
                            "(default negative, with a warning)")

    row = label.add_argument_group("row-based sequences (cannot be combined with time-based options)")
    row.add_argument("--sequence-length", type=_positive_int, default=argparse.SUPPRESS, metavar="N",
                     help="images per sequence (default: single images)")
    row.add_argument("--stride", type=_positive_int, default=argparse.SUPPRESS, metavar="N",
                     help="rows between sequence starts (default 1; needs --sequence-length)")
    row.add_argument("--cadence-minutes", type=_positive_number, default=argparse.SUPPRESS, metavar="MINUTES",
                     help="required spacing inside a sequence; other spacings are skipped (needs --sequence-length)")

    time_based = label.add_argument_group("time-based sequences")
    time_based.add_argument("--start", default=argparse.SUPPRESS, metavar="TIMESTAMP",
                            help="first sequence start; must be an image time (default: earliest image)")
    time_based.add_argument("--cadence", default=argparse.SUPPRESS, metavar="DURATION",
                            help="spacing inside a sequence, e.g. 1h")
    time_based.add_argument("--observation-window", default=argparse.SUPPRESS, metavar="DURATION",
                            help="span of one sequence, a multiple of --cadence")
    time_based.add_argument("--sliding-window", default=argparse.SUPPRESS, metavar="DURATION",
                            help="shift between sequence starts, a multiple of --cadence")
    time_based.add_argument("--report", default=argparse.SUPPRESS, metavar="REPORT.json",
                            help="also write the SequenceBuildReport (time-based only)")
    return parser


def _given(args: argparse.Namespace, name: str):
    """Value of an option whose default is SUPPRESS, or None when it was not given."""
    return getattr(args, name, None)


def _make_strategy(labeling: str, threshold):
    if labeling == "binary":
        return BinaryThresholdStrategy(threshold if threshold is not None else "M")
    if threshold is not None and labeling != "cumulative":
        raise CLIError(f"--threshold is not used by --labeling {labeling}; it applies to binary and cumulative only")
    if labeling == "max-flux":
        return MaxFlareStrategy()
    if labeling == "multiclass":
        return DatasetBuildingPipeline(MaxFluxReducer(), MulticlassLabeler())
    if labeling == "class-counts":
        return DatasetBuildingPipeline(ClassCountReducer(), ClassCountLabeler())
    if labeling == "class-presence":
        return DatasetBuildingPipeline(ClassCountReducer(), ClassPresenceLabeler())
    if labeling == "cumulative":
        if threshold is None:
            return DatasetBuildingPipeline(CumulativeClassIndexReducer(), RegressionLabeler())
        try:
            numeric = float(threshold)
        except ValueError:
            raise CLIError(
                f"--threshold for --labeling cumulative must be a number (the index is compared with it), "
                f"got {threshold!r}"
            ) from None
        return DatasetBuildingPipeline(CumulativeClassIndexReducer(), NumericThresholdLabeler(numeric))
    raise CLIError(f"unknown labeling {labeling!r}")  # unreachable: argparse restricts the choices


def _builder_kwargs(args: argparse.Namespace) -> dict:
    row_given = [n for n in _ROW_BASED if _given(args, n) is not None]
    time_given = [n for n in _TIME_BASED if _given(args, n) is not None]
    if row_given and time_given:
        raise CLIError(
            "cannot combine row-based options (" + ", ".join(_FLAG_NAMES[n] for n in row_given)
            + ") with time-based options (" + ", ".join(_FLAG_NAMES[n] for n in time_given) + ")"
        )
    if row_given and _given(args, "sequence_length") is None:
        raise CLIError("--stride and --cadence-minutes need --sequence-length (they are ignored for single images)")
    if _given(args, "report") is not None and not time_given:
        raise CLIError("--report is only produced in time-based sequence mode (--cadence/--observation-window/--sliding-window)")
    if _given(args, "missing_active_region") is not None and args.target != "active_region":
        raise CLIError("--missing-active-region only applies with --target active_region")

    kwargs = {
        "prediction_window": args.window,
        "target": args.target,
        "event_time": args.event_time,
        "interval_mode": args.interval_mode,
        # Single-image mode drops other index columns unless asked; the CLI
        # always keeps image paths and ids (sequence modes keep them anyway).
        "include_metadata": not row_given and not time_given,
    }
    if _given(args, "missing_active_region") is not None:
        kwargs["missing_active_region"] = args.missing_active_region
    for name in _ROW_BASED + _TIME_BASED:
        if _given(args, name) is not None:
            kwargs[name] = getattr(args, name)
    return kwargs


def _flagged(message: str) -> str:
    """Rewrite Python parameter names in an API error message as CLI flags."""
    for name, flag in _FLAG_NAMES.items():
        message = message.replace(name, flag)
    return message


def _json_default(value):
    if isinstance(value, pd.Timestamp):
        return None if pd.isna(value) else value.isoformat(sep=" ")
    if isinstance(value, pd.Timedelta):
        return str(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    raise TypeError(f"cannot write {type(value).__name__} as JSON")


def _encode_json_cells(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a copy in which every list/tuple/dict/array cell is JSON text."""
    out = frame.copy()
    for column in out.columns:
        if out[column].dtype != object:
            continue
        is_container = out[column].map(lambda v: isinstance(v, _JSON_CELL_TYPES))
        if is_container.any():
            out[column] = [
                json.dumps(v, default=_json_default) if isinstance(v, _JSON_CELL_TYPES) else v
                for v in out[column]
            ]
    return out


def _write_atomically(path: Path, write) -> None:
    """Write through a temp file in the same directory; nothing is left behind on failure."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile("w", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False)
    tmp_path = Path(handle.name)
    try:
        with handle:
            write(handle)
        os.replace(tmp_path, path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def _report_as_json(report) -> str:
    return json.dumps(dataclasses.asdict(report), indent=2, default=_json_default) + "\n"


def _print_warning(message, category, filename, lineno, file=None, line=None):
    print(f"flare-indexer: warning: {message}", file=sys.stderr)


def _run_label(args: argparse.Namespace) -> int:
    images = Path(args.images)
    catalog = Path(args.catalog)
    output = Path(args.output)
    report_path = Path(args.report) if _given(args, "report") is not None else None

    for path in (images, catalog):
        if not path.is_file():
            raise CLIError(f"input file not found: {path}")
    for path in (output, report_path):
        if path is not None and path.exists() and not args.overwrite:
            raise CLIError(f"output file exists: {path} (pass --overwrite to replace it)")
    if report_path is not None and report_path.resolve() == output.resolve():
        raise CLIError("--report and --output must be different files")

    try:
        strategy = _make_strategy(args.labeling, _given(args, "threshold"))
        builder = DatasetBuilder(strategy=strategy, **_builder_kwargs(args))
    except (ValueError, TypeError) as exc:
        raise CLIError(_flagged(str(exc))) from None

    with warnings.catch_warnings():
        warnings.simplefilter("always")
        warnings.showwarning = _print_warning
        try:
            result = builder.build(images, catalog, return_report=report_path is not None)
        except (ValueError, TypeError) as exc:
            raise CLIError(_flagged(str(exc))) from None
    frame, report = result if report_path is not None else (result, None)

    _write_atomically(output, lambda handle: _encode_json_cells(frame).to_csv(handle, index=False, lineterminator="\n"))
    if report is not None:
        _write_atomically(report_path, lambda handle: handle.write(_report_as_json(report)))
    print(f"flare-indexer: wrote {len(frame)} row(s) to {output}"
          + (f" and the report to {report_path}" if report is not None else ""), file=sys.stderr)
    return 0


def main(argv=None) -> int:
    """Entry point of the ``flare-indexer`` console script. Returns the exit status."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "label":
            return _run_label(args)
        parser.error(f"unknown command {args.command!r}")  # unreachable: argparse restricts the choices
    except (CLIError, OSError, pd.errors.ParserError) as exc:
        print(f"flare-indexer: error: {exc}", file=sys.stderr)
        return 1
    return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
