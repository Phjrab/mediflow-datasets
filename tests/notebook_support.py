"""Assertions shared by tests of self-contained Colab notebooks."""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def assert_notebook_sources(path: Path, notebook: dict) -> dict[str, ast.expr]:
    """Validate executable cells and exact embedded source against repository modules."""
    seen = {}
    assignments = {}
    for item in notebook["cells"]:
        if item["cell_type"] != "code":
            continue
        assert item["outputs"] == []
        source = "".join(item["source"])
        if source.startswith("%%writefile "):
            first, code = source.split("\n", 1)
            name = Path(first.split(" ", 1)[1]).stem
            compile(code, name + ".py", "exec")
            assert code == (ROOT / f"src/mediflow_datasets/{name}.py").read_text(
                encoding="utf-8"
            )
            seen[name] = hashlib.sha256(code.encode("utf-8")).hexdigest()
            continue
        python_source = "\n".join(
            line for line in source.splitlines() if not line.lstrip().startswith("%pip ")
        )
        compile(python_source, path.name, "exec")
        if "_EXPECTED_SOURCE_SHA256 = " in source:
            assignments = {
                node.targets[0].id: node.value
                for node in ast.parse(source).body
                if isinstance(node, ast.Assign)
                and isinstance(node.targets[0], ast.Name)
            }
    assert seen, path.name
    assert seen == ast.literal_eval(assignments["_EXPECTED_SOURCE_SHA256"]), path.name
    return assignments
