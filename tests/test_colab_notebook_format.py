"""Check that readable Colab cells retain the exact embedded Python modules."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
from runpy import run_path

ROOT = Path(__file__).resolve().parents[1]
_formatter = run_path(str(ROOT / "scripts" / "format_colab_notebooks.py"))
GUIDE_MARKER = _formatter["GUIDE_MARKER"]
format_notebook = _formatter["format_notebook"]
source_text = _formatter["source_text"]


def _expected_hashes(source: str) -> dict[str, str]:
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "_EXPECTED_SOURCE_SHA256"
            for target in node.targets
        ):
            return ast.literal_eval(node.value)
    return {}


def test_all_notebooks_have_guide_and_exact_readable_sources() -> None:
    notebooks = sorted((ROOT / "notebooks").glob("*.ipynb"))
    assert len(notebooks) == 20
    embedded_count = 0
    for path in notebooks:
        notebook = json.loads(path.read_text(encoding="utf-8"))
        assert notebook["nbformat"] == 4
        cells = notebook["cells"]
        assert any(GUIDE_MARKER in source_text(item) for item in cells)
        ids = [item["id"] for item in cells]
        assert len(ids) == len(set(ids))
        expected = {}
        found = {}
        for item in cells:
            source = source_text(item)
            if item["cell_type"] == "code" and source.startswith("%%writefile "):
                first, body = source.split("\n", 1)
                name = Path(first.split(" ", 1)[1]).stem
                compile(body, name + ".py", "exec")
                assert item["metadata"]["jupyter"]["source_hidden"]
                found[name] = hashlib.sha256(body.encode("utf-8")).hexdigest()
            elif item["cell_type"] == "code" and "_EXPECTED_SOURCE_SHA256 = " in source:
                expected.update(_expected_hashes(source))
        assert found == expected, path.name
        embedded_count += len(found)
    assert embedded_count == 63


def test_formatter_does_not_rewrite_finished_notebook(tmp_path: Path) -> None:
    source = ROOT / "notebooks" / "01_common_dataset_audit_colab.ipynb"
    copy = tmp_path / source.name
    copy.write_bytes(source.read_bytes())
    modified, hashes = format_notebook(copy)
    assert not modified
    assert not hashes
    assert copy.read_bytes() == source.read_bytes()
