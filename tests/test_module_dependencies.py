"""
The package is layered; this test keeps it that way.

    goes_class                 (GOES class <-> flux; no package imports)
    events, labelers           (only goes_class)
    reducers                   (goes_class, events)
    pipeline                   (events only; reducer and labeler are passed in as objects)
    builder                    (events only; reducers and labelers are passed in as objects)
    datasets                   (builder; the four dataset classes)
    loaders                    (goes_class, events; input adapters, outside the core)

A module may import only from layers below it. The allowed sets below are
exact: adding a new internal import means updating this table.
"""

import ast
from pathlib import Path

import pytest

PACKAGE = Path(__file__).resolve().parent.parent / "src" / "solix"

ALLOWED = {
    "goes_class": set(),
    "events": {"goes_class"},
    "labelers": {"goes_class"},
    "reducers": {"goes_class", "events"},
    "pipeline": {"events"},
    "builder": {"events"},
    "datasets": {"builder"},
    "loaders": {"goes_class", "events"},
}


def _internal_imports(module: str) -> set[str]:
    tree = ast.parse((PACKAGE / f"{module}.py").read_text())
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.level == 1 and node.module:
            found.add(node.module.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("solix."):
            found.add(node.module.split(".")[1])
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("solix."):
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
    assert seen["goes_class"] == 1 and seen["builder"] > seen["events"] > seen["goes_class"]
