"""
examples/real_data/fetch_real_data.py with a fake Helioviewer: discovery,
manifest replay and SHA-256 checks, without network access.
"""

import io
import sys
import urllib.parse
from pathlib import Path

import pandas as pd
import pytest

REAL_DATA = Path(__file__).resolve().parent.parent / "examples" / "real_data"
pytestmark = pytest.mark.skipif(not REAL_DATA.is_dir(), reason="examples/ not available (installed package)")

if REAL_DATA.is_dir():
    sys.path.insert(0, str(REAL_DATA))
    import fetch_real_data as fetch

REQUESTED = pd.date_range("2017-09-03 01:00:00", periods=4, freq="1h")


def _jp2(date_obs: str, salt: str = "") -> bytes:
    """Bytes with the FITS-in-XML header read_jp2_header looks for."""
    return (f"\x00\x00\x00\x0cjP  <?xml version='1.0'?><meta><fits><DATE_OBS>{date_obs}</DATE_OBS>"
            f"<INSTRUME>HMI_FRONT2</INSTRUME><CONTENT>MAGNETOGRAM</CONTENT></fits></meta>{salt}").encode()


class FakeArchive:
    """Answers getClosestImage and getJP2Image from dictionaries and logs every URL."""

    def __init__(self):
        # requested time -> (image id, Helioviewer date rounded to seconds)
        self.closest = {
            "2017-09-03T01:00:00Z": ("101", "2017-09-03 00:59:50"),
            "2017-09-03T02:00:00Z": ("102", "2017-09-03 01:59:50"),
            "2017-09-03T03:00:00Z": ("103", "2017-09-03 03:36:35"),   # too far: missing
            "2017-09-03T04:00:00Z": ("104", "2017-09-03 03:59:50"),
        }
        self.files = {
            "101": _jp2("2017-09-03T00:59:49.90Z"),
            "102": _jp2("2017-09-03T01:59:49.90Z"),
            "103": _jp2("2017-09-03T03:36:35.00Z"),
            "104": _jp2("2017-09-03T03:59:50.00Z"),
        }
        self.calls = []

    def get(self, url, timeout=120):
        self.calls.append(url)
        query = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
        if "getClosestImage" in url:
            image_id, date = self.closest[query["date"][0]]
            return (f'{{"id": "{image_id}", "date": "{date}", "width": 4096, "height": 4096}}').encode()
        if "getJP2Image" in url:
            return self.files[query["id"][0]]
        raise AssertionError(f"unexpected URL {url}")

    def closest_calls(self):
        return [u for u in self.calls if "getClosestImage" in u]

    def downloads(self):
        return [urllib.parse.parse_qs(urllib.parse.urlparse(u).query)["id"][0] for u in self.calls if "getJP2Image" in u]


@pytest.fixture
def archive(monkeypatch):
    fake = FakeArchive()
    monkeypatch.setattr(fetch, "_get", fake.get)
    monkeypatch.setattr(fetch, "REQUESTED_TIMES", REQUESTED)
    return fake


@pytest.fixture
def discovered(archive, tmp_path):
    manifest = fetch.discover(tmp_path)
    archive.calls.clear()
    return manifest


def _manifest_text(root):
    return (root / fetch.MANIFEST_NAME).read_text()


def test_discovery_records_full_precision_and_missing(discovered, tmp_path):
    assert discovered["status"].tolist() == ["ok", "ok", "missing", "ok"]
    # Full-precision DATE_OBS is the timestamp; the rounded API date is kept separately.
    assert discovered.loc[0, "timestamp"] == "2017-09-03 00:59:49.900000"
    assert discovered.loc[0, "helioviewer_date"] == "2017-09-03 00:59:50"
    assert discovered.loc[2, "image_path"] == "" and discovered.loc[2, "offset_seconds"] == 2195.0
    assert discovered.loc[3, "image_path"] == "images/hmi_magnetogram_104.jp2"
    assert sorted(p.name for p in (tmp_path / "images").iterdir()) == [
        "hmi_magnetogram_101.jp2", "hmi_magnetogram_102.jp2", "hmi_magnetogram_104.jp2",
    ]


def _close_requests(archive, monkeypatch, second_id):
    """Two requests 30 s apart, both within MAX_OFFSET of the image at 00:59:49.9."""
    monkeypatch.setattr(fetch, "REQUESTED_TIMES", pd.DatetimeIndex(["2017-09-03 01:00:00", "2017-09-03 01:00:30"]))
    archive.closest["2017-09-03T01:00:30Z"] = (second_id, "2017-09-03 00:59:50")


def test_discovery_marks_a_reused_image_id_as_duplicate(archive, monkeypatch, tmp_path):
    _close_requests(archive, monkeypatch, "101")
    manifest = fetch.discover(tmp_path)
    assert manifest["status"].tolist() == ["ok", "duplicate"]
    assert manifest.loc[1, "duplicate_of"] == "2017-09-03 01:00:00.000000"
    assert manifest.loc[1, "image_path"] == ""
    assert archive.downloads() == ["101"]


def test_discovery_marks_a_second_image_with_the_same_timestamp_as_duplicate(archive, monkeypatch, tmp_path):
    _close_requests(archive, monkeypatch, "105")
    archive.files["105"] = _jp2("2017-09-03T00:59:49.90Z", salt="other file")
    manifest = fetch.discover(tmp_path)
    assert manifest["status"].tolist() == ["ok", "duplicate"]
    assert manifest.loc[1, ["helioviewer_id", "duplicate_of"]].tolist() == ["105", "2017-09-03 01:00:00.000000"]
    assert manifest.loc[1, "image_path"] == ""

    # A replay downloads neither the missing nor the duplicate entries again.
    for image in (tmp_path / "images").iterdir():
        image.unlink()
    archive.calls.clear()
    fetch.replay(tmp_path)
    assert archive.downloads() == ["101"] and archive.closest_calls() == []


def test_replay_uses_recorded_ids_and_skips_missing(archive, discovered, tmp_path):
    before = _manifest_text(tmp_path)
    for image in (tmp_path / "images").iterdir():
        image.unlink()

    fetch.replay(tmp_path)

    assert archive.closest_calls() == []
    assert archive.downloads() == ["101", "102", "104"]  # not 103, recorded as missing
    assert pd.read_csv(tmp_path / fetch.MANIFEST_NAME, dtype=str, keep_default_na=False).equals(
        pd.read_csv(io.StringIO(before), dtype=str, keep_default_na=False))


def test_replay_with_cached_files_needs_no_network(archive, discovered, tmp_path):
    fetch.replay(tmp_path)
    assert archive.calls == []


def test_replay_stops_on_modified_cached_file(archive, discovered, tmp_path):
    before = _manifest_text(tmp_path)
    image = tmp_path / "images" / "hmi_magnetogram_104.jp2"
    image.write_bytes(_jp2("2017-09-03T03:59:50.00Z", salt="modified"))
    modified = image.read_bytes()

    with pytest.raises(fetch.HashMismatchError, match="hmi_magnetogram_104.jp2"):
        fetch.replay(tmp_path)

    assert image.read_bytes() == modified
    assert _manifest_text(tmp_path) == before
    assert archive.calls == []


def test_replay_stops_on_changed_download_without_writing_it(archive, discovered, tmp_path):
    before = _manifest_text(tmp_path)
    image = tmp_path / "images" / "hmi_magnetogram_101.jp2"
    image.unlink()
    archive.files["101"] = _jp2("2017-09-03T00:59:49.90Z", salt="archive changed")

    with pytest.raises(fetch.HashMismatchError, match="Nothing was written"):
        fetch.replay(tmp_path)

    assert not image.exists()
    assert not image.with_suffix(".jp2.part").exists()
    assert _manifest_text(tmp_path) == before


def test_refresh_does_not_replace_a_recorded_hash_for_the_same_id(archive, discovered, tmp_path):
    before = _manifest_text(tmp_path)
    (tmp_path / "images" / "hmi_magnetogram_101.jp2").unlink()
    archive.files["101"] = _jp2("2017-09-03T00:59:49.90Z", salt="archive changed")
    with pytest.raises(fetch.HashMismatchError):
        fetch.discover(tmp_path)
    assert _manifest_text(tmp_path) == before


def test_main_replays_by_default_and_discovers_only_with_refresh(archive, discovered, tmp_path):
    fetch.main([], root=tmp_path)
    assert archive.closest_calls() == []

    fetch.main(["--refresh"], root=tmp_path)
    assert len(archive.closest_calls()) == len(REQUESTED)


def test_first_acquisition_discovers(archive, tmp_path):
    fetch.main([], root=tmp_path)
    assert len(archive.closest_calls()) == len(REQUESTED)
    assert (tmp_path / fetch.MANIFEST_NAME).exists()
