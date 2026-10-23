"""
Example 2 -- exactly which flares fall inside the prediction window.

The window is always (t, t + w): a flare must start after the image
time t and before t + w; one at exactly t or exactly t + w does not count. Uses the synthetic M1.0 flare that starts at
2024-03-05 00:00 (peak 00:10) to show how the window length and
`event_time` change the answer at the edges.

Run:  python examples/02_window_boundaries.py
"""

from pathlib import Path

import pandas as pd

import flare_indexer as fidx

DATA = Path(__file__).parent / "data"


def main():
    matcher = fidx.EventMatcher(DATA / "flare_catalog.csv")

    cases = [
        # (image time t, window, event_time, description)
        ("2024-03-04 00:00:00", "24h", "start", "starts exactly at t+24h: excluded"),
        ("2024-03-04 00:00:00", "24h10min", "start", "window 10 min longer: included"),
        ("2024-03-04 00:00:00", "24h10min", "peak", "peak exactly at t+w: excluded"),
        ("2024-03-05 00:00:00", "1h", "start", "starts exactly at t: excluded"),
    ]
    for image_time, window, event_time, description in cases:
        flares = matcher.query(pd.Timestamp(image_time), window, event_time=event_time)
        found = ", ".join(f"{f.goes_class} (start {f.start_time:%m-%d %H:%M})" for f in flares) or "none"
        label = fidx.BinaryThresholdStrategy("M").label(flares)
        print(f"t={image_time}  w={window:<8}  {event_time:<5}  label={label}  {description:<36} flares: {found}")

    print(
        "\n(t, t+w) counts neither a flare that starts at the image time nor one that\n"
        "starts exactly at t+w: a 24 h window means 't < start <= t + 23:59:59'."
    )


if __name__ == "__main__":
    main()
