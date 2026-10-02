"""
Builds the architecture diagrams from one description, in two formats:

- flare_indexer_architecture.drawio  (editable in draw.io / diagrams.net,
  one tab per page)
- flare_indexer_architecture.pdf     (vector, one page per diagram)

Every box names a real module, class, function or file in this
repository; update this script when the code changes and rerun it.

Usage (needs matplotlib, which is not a package dependency):
    pip install matplotlib
    python docs/architecture/build_diagrams.py [--png PREVIEW_DIR]
"""

import argparse

from dataclasses import dataclass, field
from pathlib import Path
from xml.sax.saxutils import escape

OUT_DIR = Path(__file__).parent
PAGE_W, PAGE_H = 1169, 827  # A4 landscape at 100 px per inch

STYLES = {
    # kind: (fill, stroke)
    "input": ("#dae8fc", "#6c8ebf"),
    "loader": ("#d5e8d4", "#82b366"),
    "core": ("#fff2cc", "#d6b656"),
    "strategy": ("#ffe6cc", "#d79b00"),
    "output": ("#e1d5e7", "#9673a6"),
    "decision": ("#f8cecc", "#b85450"),
    "note": ("#ffffff", "#999999"),
    "band": ("#f5f5f5", "#cccccc"),
    "window": ("#b9e0a5", "#82b366"),
    "window_alt": ("#fad7ac", "#b46504"),
    "missing": ("#ffffff", "#b85450"),
    "text": (None, None),
}


@dataclass
class Box:
    id: str
    x: float
    y: float
    w: float
    h: float
    text: str
    kind: str = "core"
    title: bool = False  # bold first line


@dataclass
class Edge:
    src: str
    dst: str
    label: str = ""
    dashed: bool = False


@dataclass
class Page:
    name: str
    title: str
    subtitle: str
    boxes: list = field(default_factory=list)
    edges: list = field(default_factory=list)


def page_overview() -> Page:
    p = Page("1 Overview", "flare-indexer: components and data flow",
             "Real inputs -> loaders -> CSV inputs -> DatasetBuilder -> EventMatcher -> strategy -> labeled table")
    b = p.boxes
    b += [
        Box("raw_img", 30, 110, 200, 90, "HMI image list\n(headerless paths, e.g.\ntotalfiles_jpg_512.csv)", "input"),
        Box("raw_cat", 30, 400, 200, 90, "GOES catalog\n(goes_flares_integrated.csv:\nstart/peak/end_time, goes_class,\nnoaa_active_region)", "input"),
        Box("load_img", 260, 110, 250, 90, "loaders.\nbuild_image_index_from_filenames()\nparse HMI.m...jpg file names;\nstrict, duplicate_policy", "loader"),
        Box("load_cat", 260, 400, 250, 90, "loaders.adapt_goes_catalog()\nkeep full timestamps,\nnormalize noaa_active_region", "loader"),
        Box("img_rep", 260, 230, 250, 50, "ImageIndexReport\n(excluded rows, duplicates)", "output"),
        Box("idx", 540, 110, 200, 90, "Image index CSV\ntimestamp [, active_region]\n[, image_path, ...]\n(or written by hand)", "input"),
        Box("cat", 540, 400, 200, 90, "Flare catalog CSV\nstart_time, peak_time,\nend_time, class, active_region\n(or HHMM scraper schema)", "input"),
        Box("builder", 770, 100, 370, 120, "builder.DatasetBuilder\n__init__: validate parameters; choose mode (single image,\nrow-based or time-based sequences) and target\nbuild(index, catalog): read and check the index, loop over\nreference times t, collect records", "core", True),
        Box("matcher", 770, 290, 370, 110, "events.EventMatcher\nload catalog, detect schema, validate GOES classes, sort\nquery(t, prediction_window, active_region,\nevent_time, interval_mode) -> list[FlareEvent]", "core", True),
        Box("strategy", 770, 470, 180, 130, "Strategy\nlabel(flares)\nBinaryThreshold-\nStrategy(\"M\")\nMaxFlareStrategy()\nDatasetBuilding-\nPipeline", "strategy", True),
        Box("reducer", 980, 470, 160, 60, "MaxFluxReducer\nflares -> max flux", "strategy"),
        Box("labeler", 980, 560, 160, 60, "BinaryThresholdLabeler\n/ RegressionLabeler", "strategy"),
        Box("flux", 980, 650, 160, 60, "FluxConverter\n\"M2.3\" -> 2.3e-5 W/m^2", "core"),
        Box("out", 770, 660, 180, 120, "DataFrame\none row per image\nor sequence; label +\nrequested metadata", "output", True),
        Box("report", 540, 660, 200, 120, "Also returned/raised:\nSequenceBuildReport\n(time-based mode)\nUserWarning (missing\nactive regions)", "output"),
    ]
    p.edges += [
        Edge("raw_img", "load_img"), Edge("raw_cat", "load_cat"),
        Edge("load_img", "idx", "writes"), Edge("load_cat", "cat", "writes"), Edge("load_img", "img_rep"),
        Edge("idx", "builder"), Edge("cat", "matcher"),
        Edge("builder", "matcher", "each t"), Edge("matcher", "strategy", "flares"),
        Edge("strategy", "reducer"), Edge("reducer", "labeler", "value"), Edge("labeler", "flux", "uses", dashed=True),
        Edge("strategy", "out", "label"), Edge("out", "report", "", dashed=True),
    ]
    return p


def page_build_flow() -> Page:
    p = Page("2 build() flow", "DatasetBuilder.build(): control flow",
             "Mode is fixed in __init__; build() dispatches to one of six private builders (three modes x two targets)")
    b = p.boxes
    b += [
        Box("read", 40, 100, 250, 60, "pd.read_csv(image_index)\nempty file -> ValueError", "core"),
        Box("cols", 40, 190, 250, 70, "require timestamp; require\nactive_region if\ntarget=\"active_region\"", "decision"),
        Box("em", 40, 290, 250, 60, "EventMatcher(catalog)\nschema + GOES class check, sort", "core"),
        Box("warn", 40, 380, 250, 70, "AR mode: warn if catalog\nflares have no active_region", "decision"),
        Box("mode", 330, 250, 180, 90, "Which mode?\ntime-based params?\nsequence_length > 1?", "decision", True),
        Box("single", 560, 100, 270, 110, "Single image\n_build_single_image[_active_region]\none record per index row\nt = image time", "core", True),
        Box("rowseq", 560, 240, 270, 130, "Row-based sequences\nindex sorted + unique, else error\nrows[i : i+length], i += stride\nskip if cadence_minutes violated\nt = last image", "core", True),
        Box("timeseq", 560, 400, 270, 160, "Time-based sequences\nunique timestamps; infer source\ncadence (most common gap); grid\nstart + k*cadence; skip if any grid\ntime missing; start += sliding_window\nt = last image; SequenceBuildReport", "core", True),
        Box("query", 880, 240, 250, 110, "For every reference t:\nmatcher.query(t,\nprediction_window,\nactive_region, event_time,\ninterval_mode)", "core", True),
        Box("label", 880, 400, 250, 60, "strategy.label(flares) -> label", "strategy"),
        Box("df", 880, 500, 250, 60, "pd.DataFrame(records)[columns]", "output"),
        Box("ar", 40, 600, 790, 140, "target=\"active_region\" variants of every mode\n"
            "normalize region (13600 / 13600.0 / \"13600\" -> \"13600\"); query only that region's flares\n"
            "sequences: group rows by region first (_region_groups); checks and grids run per region\n"
            "image region missing -> missing_active_region: \"negative\" (+UserWarning, own group) / \"drop\" / \"error\"\n"
            "time-based: source cadence from all images; one grid per region; report.region_count",
            "decision", True),
    ]
    p.edges += [
        Edge("read", "cols"), Edge("cols", "em"), Edge("em", "warn"), Edge("warn", "mode"),
        Edge("mode", "single"), Edge("mode", "rowseq"), Edge("mode", "timeseq"),
        Edge("single", "query"), Edge("rowseq", "query"), Edge("timeseq", "query"),
        Edge("query", "label"), Edge("label", "df"),
    ]
    return p


def page_windows() -> Page:
    p = Page("3 Time windows", "Prediction window semantics",
             "Which flares count for an image at time t (w = prediction_window). Times are compared as given (no TAI/UTC conversion).")
    b = p.boxes
    x0, scale = 520, 11  # px per hour, t at x0
    axis_y = 140

    def hx(hours):
        return x0 + hours * scale

    b += [
        Box("axis", hx(-26), axis_y, 52 * scale, 3, "", "band"),
        Box("tm24", hx(-24) - 22, axis_y - 32, 70, 24, "t - 24h", "text"),
        Box("t", hx(0) - 8, axis_y - 32, 30, 24, "t", "text"),
        Box("t24", hx(24) - 22, axis_y - 32, 70, 24, "t + 24h", "text"),
        Box("legend", 30, axis_y + 12, 220, 24, "filled dot = included, ring = excluded", "text"),
    ]
    rows = [
        ("left", "[t, t+w)", "interval_mode=\"left_closed\" (package default)\nincludes t, excludes t + w", 0, 24, True, False, "window"),
        ("right", "(t, t+w]", "interval_mode=\"right_closed\" (lab full-disk)\nexcludes t, includes t + w", 0, 24, False, True, "window"),
        ("legacy", "(t, t+23:59:59]", "prediction_window=\"23h59min59s\"\nreproduces labeling.py (full disk) exactly", 0, 24 - 1 / 3600, False, True, "window"),
        ("lookback", "[t-24h, t]", "published arpatch-sfp AR labels\n(step3_labeling.py); NOT supported, see D2", -24, 0, True, True, "window_alt"),
    ]
    y = 200
    for rid, name, desc, start, end, closed_left, closed_right, kind in rows:
        b.append(Box(f"{rid}_txt", 30, y, 210, 30, name, "text"))
        b.append(Box(f"{rid}_bar", hx(start), y + 4, (end - start) * scale, 22, "", kind))
        b.append(Box(f"{rid}_l", hx(start) - 9, y, 20, 30, "\u25cf" if closed_left else "\u25cb", "text"))
        b.append(Box(f"{rid}_r", hx(end) - 9, y, 20, 30, "\u25cf" if closed_right else "\u25cb", "text"))
        b.append(Box(f"{rid}_desc", hx(26), y - 6, 1140 - hx(26), 42, desc, "note"))
        y += 70
    b += [
        Box("ev", 30, 500, 520, 110,
            "event_time selects the flare time that must fall inside the window:\n"
            "  \"peak\" (default) -> peak_time\n  \"start\" -> start_time (lab convention)\n"
            "Sequences: t = sequence_end (last image of the sequence).", "note"),
        Box("ex", 590, 500, 550, 130,
            "Example from the real full-disk data (2010-2018, 63,649 images):\n"
            "image 2014-08-23 12:00, M5.9 flare starts 2014-08-24 12:00 (= t + 24h)\n"
            "  (t, t+24h]       -> label 1\n  (t, t+23:59:59]  -> label 0 (legacy)\n"
            "Six images differ this way; with 23h59min59s all 63,649 agree.", "decision"),
    ]
    return p


def page_sequences() -> Page:
    p = Page("4 Sequences", "Sequence construction",
             "Row-based uses positions in the index; time-based uses an exact timestamp grid and skips incomplete grids")
    b = p.boxes
    b.append(Box("rb_title", 30, 95, 1100, 30,
                 "Row-based: sequence_length=3, stride=3, cadence_minutes=60 (index must be sorted and unique)", "text"))
    hours = [0, 1, 2, 3, 4, 5, 8, 9, 10]
    for i, h in enumerate(hours):
        b.append(Box(f"rb{i}", 60 + i * 110, 130, 90, 40, f"row {i}\n{h:02d}:00", "input"))
    b += [
        Box("rbs1", 60, 180, 310, 28, "seq 1: rows 0-2, t = 02:00", "window"),
        Box("rbs2", 390, 180, 310, 28, "seq 2: rows 3-5, t = 05:00", "window"),
        Box("rbs3", 720, 180, 310, 28, "seq 3: rows 6-8: 05:00 -> 08:00 gap -> skipped", "decision"),
        Box("rbnote", 60, 218, 970, 32,
            "Without cadence_minutes, seq 3 would be kept although its images are not 60 min apart.", "note"),
    ]
    b.append(Box("tb_title", 30, 275, 1100, 30,
                 "Time-based: cadence=\"2h\", observation_window=\"6h\" (3 images), sliding_window=\"2h\"; inferred source cadence = 1h", "text"))
    cell = 88
    for i in range(12):
        missing = i in (6, 7)
        b.append(Box(f"tb{i}", 60 + i * cell, 310, 76, 40, f"{i:02d}:00" + ("\nmissing" if missing else ""),
                     "missing" if missing else "input"))
    grid = [
        ("g1", 0, "00:00 -> 00, 02, 04: emitted, t = 04:00", "window"),
        ("g2", 2, "02:00 -> 02, 04, 06: 06:00 missing, skipped", "decision"),
        ("g3", 4, "04:00 -> 04, 06, 08: skipped", "decision"),
        ("g4", 6, "06:00 -> 06, 08, 10: skipped", "decision"),
        ("g5", 8, "08:00 -> needs 12:00 > last image: stop", "note"),
    ]
    for j, (gid, start, text, kind) in enumerate(grid):
        x = 60 + start * cell
        b.append(Box(gid, x, 362 + j * 38, min(4 * cell + 76, 1140 - x), 30, text, kind))
    b += [
        Box("rules", 60, 570, 560, 180,
            "Rules (builder._grid_candidates and its callers)\n"
            "cadence >= inferred cadence and an exact multiple of it\n"
            "observation_window, sliding_window: multiples of cadence\n"
            "start (optional) must be an existing image time\n"
            "candidates while start + observation_window - cadence <= last image\n"
            "missing grid time -> skip; never interpolate or substitute\n"
            "return_report=True -> SequenceBuildReport (counts, first 50 skips)",
            "note", True),
        Box("out", 650, 570, 480, 90,
            "Output row per emitted sequence\nsequence_start, sequence_end, timestamps (list), n_images,\n"
            "[active_region], label, + every other index column as a list", "output", True),
        Box("arseq", 650, 675, 480, 75,
            "target=\"active_region\": all of the above runs per region;\n"
            "regions may share timestamps; a gap affects only its own region", "decision", True),
    ]
    return p


def page_active_region() -> Page:
    p = Page("5 Active regions", "Active-region matching and missing data",
             "target=\"active_region\": a flare counts only if its region key equals the patch's region key")
    b = p.boxes
    b += [
        Box("patch", 40, 110, 230, 80, "Patch row\ntimestamp, active_region,\nimage_path", "input"),
        Box("norm", 320, 110, 230, 80, "_normalize_active_region()\n13600 / 13600.0 / \"13600\" -> \"13600\"\nblank -> None", "core"),
        Box("missing_q", 600, 100, 230, 100, "image region\nmissing?", "decision", True),
        Box("policy", 880, 90, 250, 130, "missing_active_region\n\"negative\" (default): label as no\nflare + UserWarning\n\"drop\": leave out\n\"error\": ValueError", "decision", True),
        Box("query", 600, 260, 230, 90, "EventMatcher.query(t, w,\nactive_region=region)\ncandidates in window, then\nfilter catalog region == region", "core"),
        Box("catmiss", 880, 260, 250, 90, "Catalog flare without region:\nnever matches any patch\n(UserWarning with count)", "decision"),
        Box("seq", 40, 260, 510, 90, "Sequences: rows grouped by region first; each region's sequences\nuse only its images; unique timestamps required within a region.\nRows without a region never join a region's sequence.", "note"),
        Box("label", 600, 400, 230, 60, "strategy.label(flares)", "strategy"),
        Box("mapping", 40, 400, 510, 150,
            "Region keys in the lab data\n"
            "patch: SHARP filename -> HARP number (e.g. 362)\n"
            "flare: goes_flares_integrated.csv -> NOAA number (e.g. 11153)\n"
            "allharp2noaa.csv: many-to-many (<= 6 NOAA per HARP, <= 2 HARP per NOAA)\n"
            "-> build a HARP-keyed catalog (scripts/validate_active_region.py)", "input", True),
        Box("facts", 600, 500, 530, 230,
            "Published-label facts (lab catalog + arpatch-sfp folds)\n"
            "- 2,343 of 14,401 flares have no NOAA region (33 of 792 M/X)\n"
            "- 108 flares have a NOAA region with no HARP\n"
            "- 511,421 published patches (-90..90 folds); up to 22 per timestamp\n"
            "- look-back [t-1 day, t] reproduces label + goes_class: 100 %\n"
            "- package forward (t, t+24h]: 97.573 % label agreement\n"
            "- catalog coverage is not stated in the catalog\n"
            "Each gap can cause false negatives; affected rows are unknown.", "decision", True),
    ]
    p.edges += [
        Edge("patch", "norm"), Edge("norm", "missing_q"),
        Edge("missing_q", "policy", "yes"), Edge("missing_q", "query", "no"),
        Edge("catmiss", "query", "", dashed=True), Edge("query", "label"),
    ]
    return p


def page_modules() -> Page:
    p = Page("6 Modules and tests", "Package modules, dependencies and verification",
             "src/flare_indexer/* (public API re-exported from __init__.py); arrows point from user to dependency")
    b = p.boxes
    b += [
        Box("init", 460, 90, 250, 60, "__init__.py\npublic API (19 names)", "output"),
        Box("builder", 60, 200, 250, 80, "builder.py\nDatasetBuilder, SequenceBuildReport", "core"),
        Box("pipeline", 360, 200, 250, 80, "pipeline.py\nDatasetBuildingPipeline,\nPipelineResult", "core"),
        Box("strategies", 640, 200, 220, 80, "strategies.py\nBinaryThresholdStrategy,\nMaxFlareStrategy", "strategy"),
        Box("loaders", 885, 200, 255, 80, "loaders.py\nbuild_image_index_from_filenames,\nadapt_goes_catalog", "loader"),
        Box("events", 160, 360, 300, 80, "events.py\nEventMatcher, FlareEvent,\n_parse_prediction_window,\n_normalize_active_region", "core"),
        Box("labelers", 560, 360, 200, 70, "labelers.py\nBinaryThresholdLabeler,\nRegressionLabeler", "strategy"),
        Box("reducers", 800, 360, 200, 70, "reducers.py\nMaxFluxReducer", "strategy"),
        Box("classifier", 640, 500, 280, 70, "classifier.py\nFluxConverter, FlareClassifier,\nInvalidGOESClassError, parse_goes_class", "core"),
        Box("deps", 160, 500, 300, 60, "pandas, numpy (only runtime deps)", "band"),
        Box("scripts", 40, 620, 520, 120,
            "Scripts, examples, committed results\n"
            "scripts/validate_full_disk.py -> real full-disk data vs legacy labels\n"
            "scripts/validate_active_region.py -> published AR labels (HARP key)\n"
            "examples/01..06_*.py + examples/data (synthetic)\n"
            "docs/validation/*.json, *.txt (committed results)", "input", True),
        Box("tests", 600, 620, 530, 120,
            "tests/ (pytest, CI on Python 3.10-3.13)\n"
            "unit: events, builder, loaders, strategies, pipeline, ...\n"
            "test_windows_and_missing_regions.py: duration windows, AR policy\n"
            "test_active_region_grouping.py, test_goes_class_validation.py\n"
            "test_examples.py: every example runs; sample data regenerates", "output", True),
    ]
    p.edges += [
        Edge("init", "builder"), Edge("init", "pipeline"), Edge("init", "strategies"), Edge("init", "loaders"),
        Edge("builder", "events"), Edge("pipeline", "events"), Edge("loaders", "events"),
        Edge("strategies", "labelers"), Edge("strategies", "reducers"),
        Edge("labelers", "classifier"), Edge("reducers", "classifier"), Edge("events", "classifier", "validates classes", dashed=True), Edge("events", "deps", dashed=True),
    ]
    return p


PAGES = [page_overview(), page_build_flow(), page_windows(), page_sequences(), page_active_region(), page_modules()]


# ── draw.io output ───────────────────────────────────────────────────────


def _drawio_value(box: Box) -> str:
    lines = [escape(line) for line in box.text.split("\n")]
    if box.title and lines:
        lines[0] = f"<b>{lines[0]}</b>"
    return "<br>".join(lines)


def write_drawio(path: Path) -> None:
    diagrams = []
    for n, page in enumerate(PAGES):
        cells = ['<mxCell id="0"/>', '<mxCell id="1" parent="0"/>']
        title = f"<b>{escape(page.title)}</b><br><i>{escape(page.subtitle)}</i>"
        cells.append(
            f'<mxCell id="title" value="{escape(title, {chr(34): "&quot;"})}" '
            'style="text;html=1;align=left;verticalAlign=top;fontSize=16;" vertex="1" parent="1">'
            f'<mxGeometry x="30" y="20" width="{PAGE_W - 60}" height="60" as="geometry"/></mxCell>'
        )
        for box in page.boxes:
            fill, stroke = STYLES[box.kind]
            if box.kind == "text":
                style = "text;html=1;align=left;verticalAlign=middle;fontSize=12;"
            else:
                dashed = "dashed=1;" if box.kind in ("note", "missing") else ""
                style = (f"rounded=1;whiteSpace=wrap;html=1;fillColor={fill};strokeColor={stroke};"
                         f"fontSize=11;align=left;spacingLeft=6;{dashed}")
            value = escape(_drawio_value(box), {chr(34): "&quot;"})
            cells.append(
                f'<mxCell id="{box.id}" value="{value}" style="{style}" vertex="1" parent="1">'
                f'<mxGeometry x="{box.x:.1f}" y="{box.y:.1f}" width="{box.w:.1f}" height="{box.h:.1f}" as="geometry"/>'
                "</mxCell>"
            )
        for i, edge in enumerate(page.edges):
            dashed = "dashed=1;" if edge.dashed else ""
            cells.append(
                f'<mxCell id="e{i}" value="{escape(edge.label)}" '
                f'style="endArrow=block;html=1;rounded=1;fontSize=10;{dashed}" edge="1" parent="1" '
                f'source="{edge.src}" target="{edge.dst}"><mxGeometry relative="1" as="geometry"/></mxCell>'
            )
        diagrams.append(
            f'<diagram id="page{n + 1}" name="{escape(page.name)}">'
            f'<mxGraphModel dx="{PAGE_W}" dy="{PAGE_H}" grid="1" gridSize="10" page="1" '
            f'pageWidth="{PAGE_W}" pageHeight="{PAGE_H}"><root>{"".join(cells)}</root></mxGraphModel></diagram>'
        )
    path.write_text('<mxfile host="flare-indexer build_diagrams.py">' + "".join(diagrams) + "</mxfile>\n")


# ── PDF output ───────────────────────────────────────────────────────────


def write_pdf(path: Path, png_dir: Path | None = None) -> None:
    import matplotlib

    matplotlib.use("pdf")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    matplotlib.rcParams["pdf.fonttype"] = 42  # embed TrueType: text stays selectable

    with PdfPages(path, metadata={"Title": "flare-indexer architecture", "Author": "flare-indexer"}) as pdf:
        for n, page in enumerate(PAGES):
            fig = plt.figure(figsize=(PAGE_W / 100, PAGE_H / 100))
            ax = fig.add_axes([0, 0, 1, 1])
            ax.set_xlim(0, PAGE_W)
            ax.set_ylim(PAGE_H, 0)
            ax.axis("off")
            ax.text(30, 30, page.title, fontsize=16, fontweight="bold", va="top")
            ax.text(30, 58, page.subtitle, fontsize=10, style="italic", va="top", color="#444444")
            ax.text(PAGE_W - 30, PAGE_H - 15, f"{n + 1} / {len(PAGES)}", fontsize=8, ha="right", color="#777777")

            patches = {}
            for box in page.boxes:
                fill, stroke = STYLES[box.kind]
                if box.kind != "text":
                    patch = FancyBboxPatch(
                        (box.x, box.y), box.w, box.h, boxstyle="round,pad=0,rounding_size=6",
                        facecolor=fill, edgecolor=stroke, linewidth=1,
                        linestyle="--" if box.kind in ("note", "missing") else "-",
                    )
                    ax.add_patch(patch)
                    patches[box.id] = patch
                if not box.text:
                    continue
                lines = box.text.split("\n")
                if box.title:
                    ax.text(box.x + 6, box.y + 6, lines[0], fontsize=9.5, fontweight="bold", va="top", family="DejaVu Sans")
                    body = "\n".join(lines[1:])
                    ax.text(box.x + 6, box.y + 22, body, fontsize=8.5, va="top", family="DejaVu Sans", linespacing=1.25)
                else:
                    ax.text(box.x + 6, box.y + box.h / 2, box.text, fontsize=8.5 if box.kind != "text" else 10,
                            va="center", family="DejaVu Sans", linespacing=1.25)

            for edge in page.edges:
                a, c = patches[edge.src], patches[edge.dst]
                sa = next(bx for bx in page.boxes if bx.id == edge.src)
                sc = next(bx for bx in page.boxes if bx.id == edge.dst)
                start = (sa.x + sa.w / 2, sa.y + sa.h / 2)
                end = (sc.x + sc.w / 2, sc.y + sc.h / 2)
                arrow = FancyArrowPatch(start, end, patchA=a, patchB=c, arrowstyle="-|>", mutation_scale=12,
                                        linewidth=1, color="#333333", linestyle="--" if edge.dashed else "-")
                ax.add_patch(arrow)
                if edge.label:
                    ax.text((start[0] + end[0]) / 2, (start[1] + end[1]) / 2, edge.label, fontsize=7.5,
                            ha="center", va="center", color="#333333",
                            bbox={"facecolor": "white", "edgecolor": "none", "pad": 1})
            pdf.savefig(fig)
            if png_dir is not None:
                fig.savefig(png_dir / f"page{n + 1}.png", dpi=110)
            plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Build the architecture diagrams.")
    parser.add_argument("--png", type=Path, default=None, help="Also write PNG previews to this directory.")
    args = parser.parse_args()
    if args.png:
        args.png.mkdir(parents=True, exist_ok=True)
    write_drawio(OUT_DIR / "flare_indexer_architecture.drawio")
    write_pdf(OUT_DIR / "flare_indexer_architecture.pdf", args.png)
    print(f"Wrote {len(PAGES)} pages to {OUT_DIR}")


if __name__ == "__main__":
    main()