# Test fixtures

All files are invented data, not solar observations. Catalogs use the
package's only schema: `start_time, peak_time, end_time, class,
active_region`, every time with its full date.

| File | Used by | Contents |
|---|---|---|
| `catalog.csv` | `test_end_to_end.py`, `test_pipeline.py`, `test_cumulative_index.py` | 4 flares in Feb 2024, regions 11111 and 22222; the X1.0 starts and peaks exactly on a 24 h window end |
| `index.csv` | `test_end_to_end.py`, `test_pipeline.py`, `test_cumulative_index.py` | 3 timestamps |
| `synthetic_same_day_catalog.csv` | `test_end_to_end.py` | 2 flares in 2030: an ordinary one and one whose start, peak and end are the same minute (equal times are allowed) |
| `synthetic_full_timestamp_catalog.csv` | `test_end_to_end.py` | 1 flare that starts 2030-01-11 23:50 and peaks 2030-01-12 00:05, i.e. crosses midnight |
