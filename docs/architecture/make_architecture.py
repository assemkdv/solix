"""Figure 1: software architecture of SOLIX.

Run:  python make_architecture.py   ->  architecture.pdf, architecture.png
"""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["pdf.fonttype"] = 42   # embed TrueType fonts; journals reject Type 3
MONO = "DejaVu Sans Mono"

W, H = 100, 104
fig = plt.figure(figsize=(6.5, 6.5 * H / W))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.axis("off")

BLUE = ("#e8f0fb", "#3b6fb6")
YELLOW = ("#fff6d6", "#b8860b")
GREY = ("#f4f4f4", "#555555")
GREEN = ("#eaf5ea", "#3a7d44")
RED = ("#fdeaea", "#b03a2e")
DARK = "#333333"


def box(x, y, w, h, colors, lw=1.1, ls="-", fc=None):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.2",
                                fc=fc or colors[0], ec=colors[1], lw=lw, ls=ls))


def text(x, y, s, size=8, ha="center", va="center", **kw):
    ax.text(x, y, s, fontsize=size, ha=ha, va=va, linespacing=1.35, **kw)


def arrow(x1, y1, x2, y2, color=DARK, label=None, lx=None, ly=None):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=11,
                                 color=color, lw=1.2, shrinkA=0, shrinkB=0))
    if label:
        text(lx if lx is not None else (x1 + x2) / 2 + 1,
             ly if ly is not None else (y1 + y2) / 2, label, size=6.8, ha="left",
             style="italic", color="#555555")


def band(y, label):
    text(1, y, label, size=7, ha="left", weight="bold", color="#777777")


# ============================ INPUT =========================================
band(101.5, "INPUT")

# input preparation: helpers that make the two files from raw data (optional)
box(2, 86, 60, 13, BLUE, ls="--", fc="white")
text(32, 95.4, "Input preparation (optional)", size=8.5, weight="bold")
text(32, 90.4, "loaders.py:  build_index_from_hmi_filenames()\n"
     "adapt_goes_catalog()   add_relative_flux()", size=6.6, family=MONO, va="center")
# (two lines: shift so they sit inside the box)

# two CSV files
box(2, 66, 29, 16, BLUE)
text(16.5, 79, "index.csv", size=8.5, weight="bold", family=MONO)
text(16.5, 72.5, "timestamp\n[active_region]\n[file_path …]", size=7, family=MONO)

box(33, 66, 29, 16, BLUE)
text(47.5, 79, "flare_catalog.csv", size=8.5, weight="bold", family=MONO)
text(47.5, 72.5, "start_time  peak_time\nend_time  class\nactive_region  [rxfi]",
     size=7, family=MONO)

arrow(16.5, 86, 16.5, 82)
arrow(47.5, 86, 47.5, 82)
text(4, 64.3, "[ ] optional column", size=6.5, color="#666666", style="italic", ha="left")

# configuration
box(66, 66, 32, 33, YELLOW)
text(82, 96, "Configuration", size=8.5, weight="bold")
text(82, 92.6, "one dataset class:", size=7, style="italic", color="#7a5c00")
text(82, 86.4, "FullDiskSingleInstance\nFullDiskSequence\nActiveRegionSingleInstance\n"
     "ActiveRegionSequence", size=6.8, family=MONO)
text(82, 80.2, "parameters:", size=7, style="italic", color="#7a5c00")
text(82, 72.6, "prediction_window\nreducer  labeler\ncadence  start\nobservation_window\n"
     "sliding_window\nmissing_active_region", size=6.6, family=MONO)

# ============================ CORE ==========================================
band(61.5, "PROCESSING")
box(2, 17, 96, 43, GREY, lw=1.3, fc="#fafafa")
text(50, 57, "dataset.build(index.csv, flare_catalog.csv)", size=8.5,
     family=MONO, weight="bold")

arrow(32, 64, 32, 60)            # files into build()
arrow(82, 66, 82, 60)            # configuration into build()

# top row: three steps
sy, sh, sw = 41, 12, 28
xs = [5, 36, 67]
titles = ["1  Validate inputs", "2  Make samples", "3  Match flares"]
bodies = ["columns, times, GOES classes\nerror on invalid rows",
          "single instance or sequence\nfrom the index",
          "EventMatcher.query(t, w, region)\nflares starting in [t, t+w)"]
for x, t, b in zip(xs, titles, bodies):
    box(x, sy, sw, sh, ("white", GREY[1]))
    text(x + sw / 2, sy + sh - 3.2, t, size=8, weight="bold")
    text(x + sw / 2, sy + 4, b, size=6.1 if "Event" in b else 6.6, family=MONO if "Event" in b else None)
arrow(xs[0] + sw, sy + sh / 2, xs[1], sy + sh / 2)
arrow(xs[1] + sw, sy + sh / 2, xs[2], sy + sh / 2)

# bottom row: the reducer, then the labeler
box(5, 19, 90, 18, GREEN, lw=1.1)
text(8, 34.6, "4  label = labeler.assign(reducer.reduce(flares))", size=8, weight="bold", ha="left")

box(8, 23.2, 38, 9, ("white", GREEN[1]), lw=0.9)
text(27, 30.4, "Reducer   flares → value", size=7.5, weight="bold")
text(27, 26.1, "MaxFlux  MinFlux  MeanFlux  ClassCount\nCumulativeClassIndex  "
     "MaxRelativeFlux\nSumRelativeFlux", size=6, family=MONO)

box(54, 23.2, 38, 9, ("white", GREEN[1]), lw=0.9)
text(73, 30.4, "Labeler   value → label", size=7.5, weight="bold")
text(73, 26.1, "BinaryThreshold  NumericThreshold\nRegression  Multiclass\n"
     "ClassCount  ClassPresence", size=6, family=MONO)
text(50, 20.9, "a reducer and a labeler are chosen separately; user-defined reducers and labelers are supported",
     size=6.4, style="italic", color="#2d5f34")
arrow(46, 27.7, 54, 27.7)

# flares go down into the reducer
arrow(xs[2] + sw / 2, sy, xs[2] + sw / 2, 37, label="flares in window")
text(50, 38.8, "repeated for each sample", size=6.6, style="italic", color="#666666")

# ============================ OUTPUT ========================================
band(14.5, "OUTPUT")
arrow(50, 17, 50, 13.5)
box(2, 1, 96, 12.5, RED)
text(50, 10.6, "pandas DataFrame, one row per sample", size=8.5, weight="bold")
text(50, 6.6, "single instance:  timestamp  [active_region]  label  [file_path …]",
     size=6.8, family=MONO)
text(50, 3.4, "sequence:  sequence_start  sequence_end  timestamps  n_instances  "
     "[active_region]  label", size=6.8, family=MONO)

fig.savefig("architecture.pdf")
fig.savefig("architecture.png", dpi=200)
