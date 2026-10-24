"""
Architecture figure of flare-indexer (docs/architecture/overview.pdf, .svg).

Regenerate from the repository root:

    python docs/architecture/make_overview.py            # writes overview.pdf, overview.svg
    python docs/architecture/make_overview.py --png      # also a PNG preview (not kept in the repository)
    python docs/architecture/make_overview.py --out DIR  # write somewhere else

Dependency: matplotlib (any 3.x; tested with 3.10). It is not a package
dependency; install it with `pip install matplotlib`. Nothing else is needed:
the boxes are placed by hand in the NODES table below and drawn with
matplotlib, so no Graphviz is required.

Box kinds: "data" (external files/tables), "pkg" (package component, module
in parentheses), "entry" (package entry points), "ext" (framework examples,
outside the package). When the code changes, edit NODES/EDGES and rerun.
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["svg.fonttype"] = "none"   # keep text as text in the SVG (editable)
matplotlib.rcParams["pdf.fonttype"] = 42       # embed TrueType so PDF text stays selectable
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

HERE = Path(__file__).resolve().parent   # docs/architecture, the default output directory
W, H = 118, 64   # drawing grid, origin bottom-left

# name: (kind, title, body lines, x, y, w, h)
NODES = {
    "raw": ("data", "Raw sources  (optional)",
            ["HMI / SHARP image file names", "GOES catalog (goes_flares_integrated)"], 2, 42, 16, 6.5),
    "loaders": ("pkg", "Input adapters  (loaders, optional)",
                ["build_image_index_from_filenames", "adapt_goes_catalog"], 2, 31, 16, 7),
    "index": ("data", "Observation index  (CSV)",
              ["timestamp", "[active_region]  [image_path, ids]"], 21, 42, 16, 6.5),
    "catalog": ("data", "Flare catalog  (CSV)",
                ["start_time, peak_time, end_time", "class, active_region"], 21, 22, 16, 6.5),
    "api": ("entry", "Python API",
            ["DatasetBuilder(window, strategy, target, ...)", ".build(index_csv, catalog_csv)"], 40, 51.5, 21, 7),
    "builder": ("pkg", "DatasetBuilder  (builder): sample construction",
                ["target:  full_disk  |  active_region  (region from the index column)",
                 "mode:    single image  |  row-based sequence  |  time-based sequence (grid, report)",
                 "per sample: reference time t = the image, or the last image of the sequence;",
                 "            window [t, t+w);  one row per image or sequence"], 40, 38, 44, 11),
    "matcher": ("pkg", "EventMatcher  (events)",
                ["loads and validates the catalog",
                 "query(t, window, region,",
                 "      event_time)  window [t, t+w)",
                 "→ list[FlareEvent] in the window"], 40, 19.5, 18, 12),
    "red_flux": ("pkg", "MaxFluxReducer  /  MinFluxReducer",
                 ["→ flux of the strongest / weakest", "   event in W/m²  (0.0 if none)"], 63, 28.5, 23, 6),
    "red_count": ("pkg", "ClassCountReducer",
                  ["→ counts per class {A, B, C, M, X}"], 63, 22, 23, 4.8),
    "red_cum": ("pkg", "CumulativeClassIndexReducer",
                ["→ Σ C + 10 Σ M + 100 Σ X",
                 "  C, M, X: GOES subclass values, not event counts",
                 "  e.g. C2.5 + M1.2 = 2.5 + 10 × 1.2 = 14.5;  A, B add 0"], 63, 13.3, 23, 7),
    "lab_flux": ("pkg", "flux  →  label",
                 ["BinaryThresholdLabeler(GOES class) → 0 / 1",
                  "RegressionLabeler → flux value",
                  "MulticlassLabeler → NF / A / B / C / M / X"], 90, 27.3, 25, 7.6),
    "lab_count": ("pkg", "counts  →  label",
                  ["ClassCountLabeler → counts dict",
                   "ClassPresenceLabeler → 0 / 1 per class"], 90, 20.8, 25, 6.4),
    "lab_cum": ("pkg", "index  →  label",
                ["NumericThresholdLabeler(number) → 0 / 1",
                 "RegressionLabeler → raw index"], 90, 14.3, 25, 6.4),
    "classifier": ("pkg", "classifier",
                   ["FluxConverter: GOES class ↔ flux (decimal)", "GOES class validation"], 40, 7.5, 18, 6.5),
    "output": ("data", "Output table  (pandas DataFrame)",
               ["single image:  timestamp [active_region] label [image_path, ids]",
                "sequence:      sequence_start, sequence_end, timestamps,",
                "               n_images [active_region] label [image_path, ids]"], 89, 50.5, 27, 8),
    "torch": ("ext", "PyTorch examples 06, 07",
              ["Dataset / DataLoader", "[B,C,H,W]  [B,T,C,H,W]"], 89, 40.5, 13, 7),
    "tf": ("ext", "TensorFlow examples 08, 09",
           ["tf.data.Dataset", "[B,H,W,C]  [B,T,H,W,C]"], 103, 40.5, 13, 7),
}

# Frame grouping reducers and labelers: the "strategy" seen by DatasetBuilder.
FRAME = (61.5, 9, 55, 27.6)
FRAME_TITLE = "strategy.label(events)  =  reducer.reduce(events)  →  labeler.assign(value)      (strategies, pipeline)"
FRAME_NOTE = ("ready-made:  BinaryThresholdStrategy = MaxFlux + BinaryThreshold;   MaxFlareStrategy = MaxFlux + Regression\n"
              "any pair:    DatasetBuildingPipeline(reducer, labeler);   any object with .label(events) is accepted as strategy")

# (source, target, label, style, source side, target side, fraction along source side, fraction along target side)
# sides: l r t b ; style: flow (solid) call (dashed) use (dotted)
EDGES = [
    ("raw", "loaders", "", "flow", "b", "t", 0.5, 0.5),
    ("loaders", "index", "writes", "flow", "r", "l", 0.7, 0.5),
    ("loaders", "catalog", "writes", "flow", "r", "l", 0.3, 0.5),
    ("index", "builder", "", "flow", "r", "l", 0.5, 0.65),
    ("catalog", "matcher", "", "flow", "r", "l", 0.5, 0.5),
    ("api", "builder", "", "flow", "b", "t", 0.5, 0.24),
    ("builder", "matcher", "t, window, region", "flow", "b", "t", 0.2, 0.5),
    ("matcher", "red_flux", "events", "flow", "r", "l", 0.8, 0.5),
    ("matcher", "red_count", "events", "flow", "r", "l", 0.5, 0.5),
    ("matcher", "red_cum", "events", "flow", "r", "l", 0.2, 0.5),
    ("red_flux", "lab_flux", "flux", "flow", "r", "l", 0.5, 0.5),
    ("red_count", "lab_count", "counts", "flow", "r", "l", 0.5, 0.5),
    ("red_cum", "lab_cum", "index", "flow", "r", "l", 0.5, 0.5),
    ("lab_flux", "builder", "label per sample", "flow", "t", "r", 0.42, 0.25),   # drawn from the frame top
    ("builder", "output", "labeled rows", "flow", "r", "l", 0.75, 0.5),
    ("output", "torch", "", "flow", "b", "t", 0.25, 0.5),
    ("output", "tf", "", "flow", "b", "t", 0.75, 0.5),
    ("classifier", "matcher", "class validation", "use", "t", "b", 0.5, 0.5),
    ("classifier", "red_flux", "GOES class ↔ flux", "use", "r", "l", 0.5, 0.5),  # drawn to the frame's left edge
]

STYLE = {
    "data": dict(face="#f5f5f5", edge="#666666", ls="--", lw=1.2),
    "pkg": dict(face="#e8f0fa", edge="#1f4e79", ls="-", lw=1.4),
    "entry": dict(face="#fff3d1", edge="#8a5a00", ls="-", lw=1.8),
    "ext": dict(face="#eaf6ea", edge="#2e6b2e", ls=":", lw=1.4),
}
ARROW = {"flow": dict(ls="-", color="#333333"), "call": dict(ls="--", color="#8a5a00"), "use": dict(ls=":", color="#1f4e79")}


def side_point(name, side, frac):
    _, _, _, x, y, w, h = NODES[name]
    if side == "l":
        return x, y + h * frac
    if side == "r":
        return x + w, y + h * frac
    if side == "t":
        return x + w * frac, y + h
    return x + w * frac, y


def draw(base: Path, png: bool = False):
    fig, ax = plt.subplots(figsize=(20, 10.0))
    ax.set_xlim(0, W)
    ax.set_ylim(5, H)
    ax.axis("off")

    # strategy frame behind the reducer/labeler boxes
    fx, fy, fw, fh = FRAME
    ax.add_patch(Rectangle((fx, fy), fw, fh, facecolor="#fbfcff", edgecolor="#1f4e79", linewidth=1.0, linestyle="-"))
    ax.text(fx + 0.8, fy + fh - 0.4, FRAME_TITLE, fontsize=8.3, fontweight="bold", va="top", color="#1f4e79")
    ax.text(fx + 0.8, fy + 0.6, FRAME_NOTE, fontsize=7.4, va="bottom", color="#222222", linespacing=1.4)

    for name, (kind, title, lines, x, y, w, h) in NODES.items():
        s = STYLE[kind]
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.6",
                                    facecolor=s["face"], edgecolor=s["edge"], linestyle=s["ls"], linewidth=s["lw"]))
        ax.text(x + w / 2, y + h - 0.7, title, ha="center", va="top", fontsize=9.2, fontweight="bold", color="#111111")
        ax.text(x + 0.7, y + h - 2.5, "\n".join(lines), ha="left", va="top", fontsize=7.9,
                family="DejaVu Sans Mono", color="#222222", linespacing=1.35)

    for src, dst, label, style, s_side, d_side, s_frac, d_frac in EDGES:
        p1 = side_point(src, s_side, s_frac)
        p2 = side_point(dst, d_side, d_frac)
        if src == "classifier" and dst == "red_flux":          # point at the frame, not the box
            p2 = (fx, p1[1])
        if src == "lab_flux" and dst == "builder":             # the label leaves the strategy frame
            p1 = (fx + fw * 0.7, fy + fh)
        a = ARROW[style]
        ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=12, linestyle=a["ls"],
                                     color=a["color"], linewidth=1.1, shrinkA=1, shrinkB=1))
        if label:
            mx, my = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
            ax.text(mx, my + 0.55, label, ha="center", va="bottom", fontsize=7.3, color=a["color"],
                    bbox=dict(boxstyle="round,pad=0.12", facecolor="white", edgecolor="none", alpha=0.92))

    # legend
    lx, ly = 2, 22
    for i, (kind, text) in enumerate([("pkg", "package code (module in parentheses)"),
                                      ("entry", "entry point of the package"),
                                      ("data", "external data: CSV files / tables"),
                                      ("ext", "framework examples, outside the package")]):
        s = STYLE[kind]
        yy = ly - i * 2.3
        ax.add_patch(FancyBboxPatch((lx, yy), 2.4, 1.5, boxstyle="round,pad=0.02,rounding_size=0.3",
                                    facecolor=s["face"], edgecolor=s["edge"], linestyle=s["ls"], linewidth=s["lw"]))
        ax.text(lx + 3.2, yy + 0.75, text, fontsize=7.6, va="center")
    for j, (style, text) in enumerate([("flow", "data flow"), ("use", "uses the GOES class ↔ flux conversion")]):
        a = ARROW[style]
        yy = ly - 4 * 2.3 - j * 1.9 + 0.75
        ax.plot([lx, lx + 2.4], [yy, yy], linestyle=a["ls"], color=a["color"], linewidth=1.2)
        ax.text(lx + 3.2, yy, text, fontsize=7.6, va="center")

    ax.text(2, 62.6, "flare-indexer: from an observation index and a flare catalog to labeled samples",
            fontsize=12.5, fontweight="bold", va="top")
    ax.text(2, 60.4, "Each box names the class or function that does the work. Solid blue boxes are package code; "
                     "dashed grey boxes are files or tables; dotted green boxes are example scripts outside the package.",
            fontsize=8.2, va="top", color="#444444")

    fig.subplots_adjust(left=0.003, right=0.997, top=0.997, bottom=0.003)
    meta = {"Title": "flare-indexer architecture"}
    fig.savefig(f"{base}.pdf", metadata=meta)
    fig.savefig(f"{base}.svg", metadata=meta)
    if png:
        fig.savefig(f"{base}.png", dpi=160)
    plt.close(fig)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Regenerate the flare-indexer architecture figure.")
    parser.add_argument("--out", type=Path, default=HERE, help="output directory (default: this script's directory)")
    parser.add_argument("--png", action="store_true", help="also write overview.png for a quick look")
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    draw(args.out / "overview", png=args.png)
    written = ["overview.pdf", "overview.svg"] + (["overview.png"] if args.png else [])
    print("written to", args.out, ":", ", ".join(written))


if __name__ == "__main__":
    main()
