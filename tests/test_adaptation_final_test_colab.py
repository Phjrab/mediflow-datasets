import ast
import hashlib
import importlib.util
import json
import sys
import types
import zipfile
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    ("failure", "message"),
    [("environment", "fixture environment failed"), ("gpu", "GPU")],
)
def test_colab_bootstrap_streams_failure_and_preserves_logs(
    tmp_path, monkeypatch, capsys, failure, message
):
    script = Path(__file__).resolve().parents[1] / "scripts/build_adaptation_final_test.py"
    spec = importlib.util.spec_from_file_location("final_test_builder", script)
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    my_drive = tmp_path / "drive/MyDrive"
    project = my_drive / "mediflow_Project"
    project.mkdir(parents=True)
    monkeypatch.setenv("MPLBACKEND", "module://matplotlib_inline.backend_inline")
    batch = """import os
import sys
def prepare_python(work, executor):
    assert os.environ["MPLBACKEND"] == "Agg"
    print("fixture backend Agg", flush=True)
    (work / "environment_probe.log").write_text("probe details", encoding="utf-8")
    print("fixture stdout before failure", flush=True)
    print("fixture stderr before failure", file=sys.stderr, flush=True)
"""
    batch += (
        '    raise RuntimeError("fixture environment failed")\n'
        if failure == "environment"
        else '    return sys.executable, {"gpu": []}\n'
    )
    bundle = project / builder.BUNDLE_NAME
    with zipfile.ZipFile(bundle, "x") as archive:
        archive.writestr("src/mediflow_datasets/__init__.py", "")
        archive.writestr("src/mediflow_datasets/domain_adaptation_batch.py", batch)
        archive.writestr(
            "src/mediflow_datasets/domain_adaptation_io.py", "def write_progress(*args): pass\n"
        )
    bundle_sha = hashlib.sha256(bundle.read_bytes()).hexdigest()
    plan_json = json.dumps({"bundle_sha256": bundle_sha})
    root = my_drive / "MediFlow_adaptation_runs" / bundle_sha[:16]
    for stage in (1, 2, 3):
        folder = root / f"stage{stage}"
        folder.mkdir(parents=True)
        (folder / "bundle.json").write_text("{}", encoding="utf-8")
    google, colab = types.ModuleType("google"), types.ModuleType("google.colab")
    colab.drive = types.SimpleNamespace(mount=lambda path: None)
    google.colab = colab
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.colab", colab)
    source = "".join(builder.notebook(plan_json)["cells"][1]["source"])

    class LocalContent(ast.NodeTransformer):
        def visit_Constant(self, node):
            if isinstance(node.value, str) and node.value.startswith("/content"):
                return ast.copy_location(
                    ast.Constant(str(tmp_path) + node.value[len("/content") :]), node
                )
            return node

    code = compile(
        ast.fix_missing_locations(LocalContent().visit(ast.parse(source))),
        "colab-failure-fixture",
        "exec",
    )
    with pytest.raises(RuntimeError, match="종료 코드 1"):
        exec(code, {"__name__": "__main__"})
    output = capsys.readouterr().out
    assert "fixture stdout before failure" in output
    assert "fixture stderr before failure" in output
    assert "fixture backend Agg" in output
    assert message in output
    logs = root / f"final_test_{hashlib.sha256(plan_json.encode()).hexdigest()[:12]}" / "logs"
    main = next(logs.glob("run_*.log"))
    assert message in main.read_text(encoding="utf-8")
    assert next(logs.glob("*_environment_probe.log")).read_text() == "probe details"
    assert not list(tmp_path.glob("mediflow_final_test_*"))
