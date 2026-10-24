"""
Architecture figure of flare-indexer (docs/architecture/overview.pdf, .svg).

Regenerate from the repository root:

    python docs/architecture/make_overview.py            # writes overview.pdf, overview.svg
    python docs/architecture/make_overview.py --png      # also a PNG preview (not kept in the repository)
    python docs/architecture/make_overview.py --out DIR  # write somewhere else

Dependency: matplotlib (any 3.x; tested with 3.10). It is not a package
dependency; install it with `pip install matplotlib`.

The figure has four layers, top to bottom: input, configuration, labeling
and output. Each layer is a colored band with plain boxes inside, placed by
hand in the LAYERS and BOXES tables below. When the code changes, edit the
tables and rerun.
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["svg.fonttype"] = "none"   # keep text as text in the SVG (editable)
matplotlib.rcParams["pdf.fonttype"] = 42       # embed TrueType so PDF text stays selectable
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

HERE = Path(__file__).resolve().parent   # docs/architecture, the default output directory
W, H = 120, 72   # drawing grid, origin bottom-left

# Layer colors (fill, border), the draw.io palette.
BLUE = ("#dae8fc", "#6c8ebf")
YELLOW = ("#fff2cc", "#d6b656")
GREEN = ("#d5e8d4", "#82b366")
RED = ("#f8cecc", "#b85450")
GREY = ("#f5f5f5", "#666666")

# name: (title, color, x, y, w, h)
LAYERS = {
    "input": ("Input Layer", BLUE, 30, 55, 88, 15),
    "config": ("Configuration Layer", YELLOW, 30, 37, 88, 14),
    "label": ("Labeling Layer", GREEN, 30, 19, 88, 14),
    "output": ("Output Layer", RED, 30, 2, 88, 13),
}

# name: (title, body lines, x, y, w, h, dashed)
BOXES = {
    # Raw data and hand-made CSVs, outside the package.
    "raw": ("Raw data (optional)",
            ["HMI / SHARP file names", "GOES flare catalog", "GOES 1-min X-ray flux"], 2, 60.5, 24, 8, False),
    "own": ("Or your own CSVs",
            ["image index: timestamp,", "[active_region], [image_path]", "flare catalog: start, peak,",
             "end, class, active_region"], 2, 38, 24, 10, False),

    # Input layer.
    "load_index": ("build_image_index_from_filenames", [], 33, 62.5, 26, 4, False),
    "load_catalog": ("adapt_goes_catalog", [], 62, 62.5, 25, 4, False),
    "load_rxf": ("add_relative_flux", [], 90, 62.5, 25, 4, False),
    "index_csv": ("Image index", [], 33, 57, 26, 3.2, False),
    "catalog_csv": ("Flare catalog", [], 62, 57, 25, 3.2, False),
    "rxf_col": ("+ rxfi column", [], 90, 57, 25, 3.2, False),

    # Configuration layer.
    "classes": ("Dataset class", ["FullDiskSingleImage", "FullDiskSequence", "ActiveRegionSingleImage",
                                  "ActiveRegionSequence"], 33, 38.5, 26, 9, False),
    "settings": ("Settings", ["prediction_window, event_time", "cadence, start",
                              "observation_window, sliding_window", "missing_active_region"], 62, 38.5, 27, 9, False),
    "strategy": ("Label (strategy)", ["reducer + labeler, or", "BinaryThresholdStrategy,",
                                      "MaxFlareStrategy, your own"], 92, 38.5, 23, 9, False),

    # Labeling layer.
    "sample": ("Each image or sequence", ["t = the image's time,", "or the last image's time"],
               33, 20.5, 18, 9, False),
    "matcher": ("EventMatcher", ["flares starting in [t, t + w)", "(only the image's region", "for active regions)"],
                54, 20.5, 18, 9, False),
    "reducer": ("Reducer", ["Max / Min / MeanFlux", "ClassCount", "CumulativeClassIndex",
                            "Max / SumRelativeFlux"], 75, 20.5, 19, 9, False),
    "labeler": ("Labeler", ["BinaryThreshold", "NumericThreshold", "Regression, Multiclass",
                            "ClassCount / Presence"], 97, 20.5, 18, 9, False),

    # Output layer.
    "table": ("Labeled table (pandas DataFrame)",
              ["timestamp or sequence columns, [active_region], label,", "[flares], [image_path, ...]",
               "verbose summary, SequenceBuildReport"], 33, 4, 46, 7.8, False),
    "frameworks": ("PyTorch Dataset / Keras PyDataset",
                   ["examples 06-09, outside the package"], 84, 4.8, 31, 6.2, True),
}

# (from, to, from side, to side, dashed)
EDGES = [
    ("raw", "load_index", "right", "left", False),
    ("load_index", "index_csv", "bottom", "top", False),
    ("load_catalog", "catalog_csv", "bottom", "top", False),
    ("load_rxf", "rxf_col", "bottom", "top", False),
    ("own", "classes", "right", "left", True),
    ("classes", "settings", "right", "left", False),
    ("settings", "strategy", "right", "left", False),
    ("sample", "matcher", "right", "left", False),
    ("matcher", "reducer", "right", "left", False),
    ("reducer", "labeler", "right", "left", False),
    ("table", "frameworks", "right", "left", True),
]

# Arrows between layers: (x, from y, to y, label)
LAYER_ARROWS = [
    (74, 55, 51, "build(image index, flare catalog)"),
    (74, 37, 33, "the same label rule for every sample"),
    (74, 19, 15, "one row per image or sequence"),
]


def side_point(name, side):
    _, _, x, y, w, h, _ = BOXES[name]
    return {"left": (x, y + h / 2), "right": (x + w, y + h / 2),
            "top": (x + w / 2, y + h), "bottom": (x + w / 2, y)}[side]


def draw(base: Path, png: bool = False):
    fig, ax = plt.subplots(figsize=(W / 8, H / 8))
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")
    font = "DejaVu Sans"

    for title, (fill, edge), x, y, w, h in LAYERS.values():
        ax.add_patch(Rectangle((x, y), w, h, facecolor="none", edgecolor=edge, linewidth=1.2))
        ax.add_patch(Rectangle((x, y + h - 2.4), w, 2.4, facecolor=fill, edgecolor=edge, linewidth=1.2))
        ax.text(x + w / 2, y + h - 1.2, title, ha="center", va="center", fontsize=10.5,
                fontweight="bold", family=font)

    for title, lines, x, y, w, h, dashed in BOXES.values():
        outside = title.startswith(("Raw data", "Or your own"))
        ax.add_patch(Rectangle((x, y), w, h, facecolor=GREY[0] if outside else "white",
                               edgecolor="black", linewidth=0.9, linestyle="--" if dashed else "-"))
        if lines:
            ax.text(x + w / 2, y + h - 1.0, title, ha="center", va="top", fontsize=9,
                    fontweight="bold", family=font)
            ax.text(x + w / 2, y + h - 3.0, "\n".join(lines), ha="center", va="top", fontsize=8.2,
                    family=font, linespacing=1.3)
        else:
            ax.text(x + w / 2, y + h / 2, title, ha="center", va="center", fontsize=9, family=font)

    def arrow(start, end, dashed=False):
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=10, linewidth=0.9,
                                     color="black", linestyle="--" if dashed else "-", shrinkA=0, shrinkB=0))

    for source, target, from_side, to_side, dashed in EDGES:
        arrow(side_point(source, from_side), side_point(target, to_side), dashed)

    for x, top, bottom, label in LAYER_ARROWS:
        arrow((x, top), (x, bottom))
        ax.text(x + 1.2, (top + bottom) / 2, label, ha="left", va="center", fontsize=8.5,
                style="italic", family=font)

    fig.tight_layout(pad=0.2)
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
    print("wrote " + ", ".join(str(args.out / name) for name in written))


if __name__ == "__main__":
    main()
