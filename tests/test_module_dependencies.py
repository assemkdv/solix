"""
The package is layered; this test keeps it that way.

    classifier                 (GOES class <-> flux; no package imports)
    events, labelers           (only classifier)
    reducers                   (classifier, events)
    strategies, pipeline       (events, reducers, labelers)
    builder                    (events only; strategies are passed in as objects)
    loaders                    (classifier, events; input adapters, outside the core)

A module may import only from layers below it. The allowed sets below are
exact: adding a new internal import means updating this table and the
dependency section of docs/architecture/system_design.md.
"""

import ast
from pathlib import Path

import pytest

PACKAGE = Path(__file__).resolve().parent.parent / "src" / "flare_indexer"

ALLOWED = {
    "classifier": set(),
    "events": {"classifier"},
    "labelers": {"classifier"},
    "reducers": {"classifier", "events"},
    "strategies": {"events", "labelers", "reducers"},
    "pipeline": {"events"},
    "builder": {"events"},
    "loaders": {"classifier", "events"},
}


def _internal_imports(module: str) -> set[str]:
    tree = ast.parse((PACKAGE / f"{module}.py").read_text())
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.level == 1 and node.module:
            found.add(node.module.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("flare_indexer."):
            found.add(node.module.split(".")[1])
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("flare_indexer."):
                    found.add(alias.name.split(".")[1])
    return found


def test_every_module_is_listed():
    modules = {p.stem for p in PACKAGE.glob("*.py")} - {"__init__"}
    assert modules == set(ALLOWED)


@pytest.mark.parametrize("module", sorted(ALLOWED))
def test_module_imports_only_from_lower_layers(module):
    assert _internal_imports(module) == ALLOWED[module]


def test_no_import_cycles():
    seen = {}

    def depth(module, stack=()):
        assert module not in stack, f"import cycle: {' -> '.join(stack + (module,))}"
        if module not in seen:
            seen[module] = 1 + max((depth(d, stack + (module,)) for d in ALLOWED[module]), default=0)
        return seen[module]

    for module in ALLOWED:
        depth(module)
    assert seen["classifier"] == 1 and seen["builder"] > seen["events"] > seen["classifier"]
