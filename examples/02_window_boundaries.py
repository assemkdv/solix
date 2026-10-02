"""
Example 2 -- exactly which flares fall inside the prediction window.

Uses one image at 2024-03-04 00:00 and the synthetic M1.0 flare that
starts at 2024-03-05 00:00 -- exactly 24 hours later -- to show how
`event_time`, `interval_mode` and the window length change the answer.

Run:  python examples/02_window_boundaries.py
"""

from pathlib import Path

import pandas as pd

import flare_indexer as fidx

DATA = Path(__file__).parent / "data"


def main():
    matcher = fidx.EventMatcher(DATA / "flare_catalog.csv")
    image_time = pd.Timestamp("2024-03-04 00:00:00")

    settings = [
        # (event_time, interval_mode, prediction_window, description)
        ("peak", "left_closed", 24, "package default:  peak_time in [t, t+24h)"),
        ("start", "right_closed", 24, "legacy-like:      start_time in (t, t+24h]"),
        ("start", "right_closed", "23h59min59s", "legacy exact:     start_time in (t, t+23:59:59]"),
    ]
    print(f"Image time t = {image_time}\n")
    for event_time, interval_mode, window, description in settings:
        flares = matcher.query(image_time, window, event_time=event_time, interval_mode=interval_mode)
        found = ", ".join(f"{f.goes_class} (start {f.start_time:%m-%d %H:%M})" for f in flares) or "none"
        label = fidx.BinaryThresholdStrategy("M").label(flares)
        print(f"{description:<50} label={label}  flares: {found}")

    print(
        "\nThe M1.0 flare starts at exactly t + 24h. It is inside (t, t+24h] but outside\n"
        "[t, t+24h) and outside (t, t+23:59:59]. On the real full-disk data this single\n"
        "boundary choice explains all 6 differences from the legacy labels."
    )


if __name__ == "__main__":
    main()
