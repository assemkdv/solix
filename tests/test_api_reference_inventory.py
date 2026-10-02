"""
docs/api_reference.md must account for every public name, every public
method of the exported classes, and every private helper (developer
appendix), so the reference cannot silently fall behind the code.
"""

import inspect
import re
from pathlib import Path

import pytest

import flare_indexer
from flare_indexer import builder, classifier, events, labelers, loaders, pipeline, reducers, strategies

DOC = Path(__file__).resolve().parent.parent / "docs" / "api_reference.md"
pytestmark = pytest.mark.skipif(not DOC.exists(), reason="documentation not available (installed package)")


def _headings():
    return set(re.findall(r"^### `([^`]+)`", DOC.read_text(), flags=re.MULTILINE))


def test_every_exported_name_has_a_section():
    missing = [name for name in flare_indexer.__all__ if name not in _headings()]
    assert not missing


def test_every_public_method_has_a_section():
    expected = []
    for name in flare_indexer.__all__:
        obj = getattr(flare_indexer, name)
        if inspect.isclass(obj) and not issubclass(obj, BaseException):
            expected += [f"{name}.{m}" for m, v in vars(obj).items() if not m.startswith("_") and callable(v)]
    missing = [m for m in expected if m not in _headings()]
    assert expected and not missing


def test_every_private_helper_is_in_the_appendix():
    appendix = DOC.read_text().split("## Developer appendix", 1)[1]
    names = []
    for module in (builder, classifier, events, labelers, loaders, pipeline, reducers, strategies):
        names += [n for n, v in vars(module).items()
                  if n.startswith("_") and not n.startswith("__") and (inspect.isfunction(v) or n.isupper())]
    for cls in (flare_indexer.DatasetBuilder, flare_indexer.EventMatcher):
        names += [f"{cls.__name__}.{m}" for m in vars(cls) if m.startswith("_") and not m.startswith("__")]
    missing = [n for n in names if f"`{n}`" not in appendix and n not in appendix]
    assert not missing
