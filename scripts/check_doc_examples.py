"""
Run the code examples embedded in the Markdown documentation and compare
their printed output with the output recorded in the same document.

An example is a fenced ```python block directly preceded by a line
`<!-- example: NAME -->`. Its recorded output is the fenced ```text block
directly preceded by `<!-- output: NAME -->`. All examples of one document
run in order, in one shared namespace (like a notebook), with the
repository root as the working directory -- so an example may use
variables defined by an earlier example of the same document; the
documents say so where that happens.

Usage (from the repository root):
    python scripts/check_doc_examples.py            # check, exit 1 on any difference
    python scripts/check_doc_examples.py --update   # rewrite recorded outputs
"""

import argparse
import contextlib
import io
import re
import sys
import traceback
import warnings
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DOCUMENTS = [
    "docs/researcher_manual.md",
    "docs/api_reference.md",
    "docs/architecture/system_design.md",
]

_BLOCK = re.compile(
    r"<!-- (?P<kind>example|output): (?P<name>[A-Za-z0-9_-]+) -->\n```(?P<lang>\w*)\n(?P<body>.*?)```",
    re.DOTALL,
)


def _run_examples(text: str, doc: str) -> dict[str, str]:
    import pandas as pd

    namespace: dict = {"__name__": "__doc_example__"}
    outputs: dict[str, str] = {}
    with pd.option_context("display.width", 120, "display.max_columns", 20):
        for match in _BLOCK.finditer(text):
            if match["kind"] != "example":
                continue
            name = match["name"]
            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer), warnings.catch_warnings():
                warnings.simplefilter("default")
                try:
                    exec(compile(match["body"], f"{doc}:{name}", "exec"), namespace)
                except Exception:
                    traceback.print_exc(file=sys.stderr)
                    raise SystemExit(f"Example {name!r} in {doc} raised an exception (see above).")
            outputs[name] = buffer.getvalue()
    return outputs


def process(doc: str, update: bool) -> list[str]:
    path = REPO_ROOT / doc
    text = path.read_text()
    outputs = _run_examples(text, doc)
    problems = []

    names_with_output = {m["name"] for m in _BLOCK.finditer(text) if m["kind"] == "output"}
    for name in names_with_output - outputs.keys():
        problems.append(f"{doc}: output block {name!r} has no example")

    def replace(match):
        if match["kind"] != "output":
            return match.group(0)
        actual = outputs.get(match["name"], "")
        if not actual.endswith("\n"):
            actual += "\n"
        if match["body"] != actual:
            problems.append(f"{doc}: output of {match['name']!r} differs")
        return f"<!-- output: {match['name']} -->\n```text\n{actual}```"

    new_text = _BLOCK.sub(replace, text)
    if update and new_text != text:
        path.write_text(new_text)
    return problems


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--update", action="store_true", help="Rewrite recorded outputs with the actual ones.")
    parser.add_argument("documents", nargs="*", default=DOCUMENTS)
    args = parser.parse_args()

    import os
    os.chdir(REPO_ROOT)
    explicit = bool(args.documents) and args.documents != DOCUMENTS
    problems = []
    checked = 0
    for doc in args.documents:
        if (REPO_ROOT / doc).exists():
            problems += process(doc, args.update)
            checked += 1
        elif explicit:
            problems.append(f"{doc}: file not found")
    if problems and not args.update:
        print("\n".join(problems))
        sys.exit(1)
    print(f"{'Updated' if args.update else 'Checked'} {checked} document(s); "
          f"{'all outputs match' if not problems else f'{len(problems)} output(s) rewritten'}.")


if __name__ == "__main__":
    main()
