"""Execute the code blocks in the README and the quickstart guide.

Documentation that does not run is documentation that is wrong, and it goes wrong silently: a reader
copies the snippet, it raises, and the package looks broken. These tests caught a real one on the first
run (the README constructed a ``Detection`` without its required ``method``), which is exactly the class
of defect they exist for.

Blocks are executed **cumulatively per file**, in document order, in one namespace. That is how a reader
follows a guide: the fleet built in the first block is the fleet scored in the third. Running each block
in isolation would fail on every guide worth writing.

Illustrative fragments that are not complete programs are marked in the source with ``no-run`` on the
fence, so the distinction is visible in the document rather than kept in this file.
"""

from __future__ import annotations

import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
RUNNABLE = ["README.md", "docs/guides/01_install-and-quickstart.md"]

FENCE = re.compile(r"^```python(?P<flags>[^\n]*)\n(?P<code>.*?)^```", re.M | re.S)


def _blocks(rel: str) -> list[str]:
    text = (ROOT / rel).read_text(encoding="utf-8")
    return [m.group("code") for m in FENCE.finditer(text) if "no-run" not in m.group("flags")]


@pytest.mark.parametrize("rel", RUNNABLE)
def test_documentation_examples_run_in_document_order(rel):
    blocks = _blocks(rel)
    assert blocks, f"{rel} has no runnable python block; did a fence get renamed?"

    namespace: dict = {"__name__": "__doc_example__"}
    for i, code in enumerate(blocks):
        try:
            exec(compile(code, f"{rel}#{i}", "exec"), namespace)
        except SystemExit as exc:
            # The quickstart documents raising on an unmet budget. That is correct behaviour to show,
            # but if it actually triggers here the documented budget has become unreachable and the
            # guide is now misleading.
            pytest.fail(f"{rel} block {i} exited: {exc}")


def test_every_runnable_file_is_covered():
    # Guards against the regex silently matching nothing after a docs reformat, which would turn this
    # whole file into a test that always passes.
    assert sum(len(_blocks(rel)) for rel in RUNNABLE) >= 4


def test_no_run_blocks_are_still_syntactically_valid():
    # A fragment excused from EXECUTION is not excused from being valid Python. A snippet that does not
    # even parse is a typo shipped to every reader.
    for rel in RUNNABLE:
        text = (ROOT / rel).read_text(encoding="utf-8")
        for m in FENCE.finditer(text):
            if "no-run" in m.group("flags"):
                compile(m.group("code"), f"{rel}#no-run", "exec")
