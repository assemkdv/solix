# Test fixtures

## Synthetic catalogs

**Invented events, not solar observations.** The dates (2030), classes and
region numbers (9000x) in the two files below were made up for
`tests/test_end_to_end.py`. Do not use them as data.

### synthetic_hhmm_catalog.csv

Legacy `date, start, peak, end, class, active_region` format: `start`,
`peak` and `end` are `HHMM` times and all three take the calendar date
from `date`.

| Row | Meaning | Parsed start / peak / end |
|---|---|---|
| M2.0, region 90001 | ordinary same-day flare | 2030-01-10 09:40 / 09:52 / 10:05 |
| C3.0, region 90003 | start, peak and end at the same minute (equal times are allowed) | 2030-01-12 12:00 / 12:00 / 12:00 |

A flare that crosses midnight cannot be written in this format: with
`2350 / 0005 / 0020` on one date, the peak and end would come before the
start. Loading such a row raises a `ValueError` naming the row; dates are
never shifted automatically. That case is built inside the test, not
stored here.

### synthetic_full_timestamp_catalog.csv

Full-timestamp `start_time, peak_time, end_time, class, active_region`
format, where each time has its own date. One flare that crosses
midnight:

| Row | start_time | peak_time | end_time |
|---|---|---|---|
| X1.5, region 90002 | 2030-01-11 23:50 | 2030-01-12 00:05 | 2030-01-12 00:20 |
