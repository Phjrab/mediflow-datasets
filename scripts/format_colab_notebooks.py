"""Make the existing Colab notebooks readable without changing embedded code."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = ROOT / "notebooks"
GUIDE_MARKER = "<!-- mediflow-notebook-guide-v1 -->"
EMBEDDED_MARKER = "<!-- mediflow-embedded-module-v1 -->"
SECTION_MARKER = "<!-- mediflow-notebook-sections-v1 -->"

PURPOSES = {
    "01": "세 도메인의 ZIP 구조·클래스·중복과 파일 오류를 학습 전에 검사합니다.",
    "02": "같은 과제에서 원본 학습과 증강 학습을 비교합니다.",
    "03": "공통 6개 실험을 실행하고 Validation 기준으로 후보를 비교합니다.",
    "04": "Hair 기준선을 여러 seed로 반복하는 과거 연구 노트북입니다.",
    "05": "Hair 논문 기반 방법을 묶어 선별하도록 준비한 과거 연구 노트북입니다.",
    "06": "Hair의 기준선과 Supervised Contrastive Learning을 비교합니다.",
    "07": "Hair SupCon의 추가 seed 반복을 위한 과거 연구 노트북입니다.",
    "08": "Hair의 DINOv2·EfficientNetV2·해상도 조건을 선별합니다.",
    "09": "Hair B1·384의 Adam과 SAM을 비교합니다.",
    "10": "고정된 Hair 5클래스 후보를 Test에서 평가하고 패키징합니다.",
    "11": "Web Skin의 WS-DAN·PMG·MixStyle을 비교합니다.",
    "12": "Web Skin PMG의 B1·384 조건을 Validation에서 비교합니다.",
    "13": "고정된 Web Skin PMG B0·256 후보를 Test에서 평가하고 패키징합니다.",
    "14": "Web Skin의 MedSigLIP frozen linear probe를 선별합니다.",
    "15": "Hair 5클래스의 MedSigLIP frozen linear probe를 선별합니다.",
    "16": "새 Hair ‘양호’ 사진과 기존 Hair 사진의 중복을 검사합니다.",
    "17": "Hair 6클래스 데이터 구성과 네 학습 조건을 실행합니다.",
    "18": "이미 선정된 Hair 6클래스 성능 우선 모델을 검증·패키징합니다.",
    "19": "이미 학습된 Hair 6클래스 경량 후보를 검증·패키징합니다.",
    "20": "저장된 34개 모델의 Colab 실행 시간을 재학습 없이 측정합니다.",
}

EXTRA_SECTIONS = {
    "16": [
        ("from google.colab import drive", "## 1. Drive 연결과 검사 대상 지정"),
    ],
    "17": [
        ("from pathlib import Path", "## 1. Drive 경로와 재개 설정"),
        ("from google.colab import drive", "## 2. Drive 연결과 실행 환경"),
    ],
    "20": [
        ("%pip -q install", "## 1. 측정 환경 설치"),
        ("from google.colab import drive", "## 2. Drive 연결과 GPU 설정"),
        ("# 학습은 하지 않으며", "## 3. 측정 설정"),
        ("# 오래 걸리는 측정 전에", "## 5. 저장 모델 사전 점검"),
        ("from datetime import datetime", "## 6. 34개 모델 측정과 결과 저장"),
    ],
}

EXTRA_RENAMES = {
    "16": {
        "## 검사 함수": "## 2. 검사 함수",
        "## 교차 검사 실행·결과 저장": "## 3. 교차 검사 실행·결과 저장",
    },
    "17": {
        "## 재현 코드": "## 3. 재현 코드",
        "## 6클래스 데이터 구성": "## 4. 6클래스 데이터 구성",
        "## 4개 실험·그래프·선정 모델 Test": "## 5. 4개 실험·그래프·선정 모델 Test",
    },
    "20": {"## 측정 코드": "## 4. 측정 코드", "## 결과 읽는 법": "## 7. 결과 읽는 법"},
}


def source_text(cell: dict) -> str:
    source = cell.get("source", [])
    return "".join(source) if isinstance(source, list) else source


def source_lines(value: str) -> list[str]:
    return value.splitlines(keepends=True)


def cell(kind: str, source: str, *, hidden: bool = False) -> dict:
    result = {"cell_type": kind, "metadata": {}, "source": source_lines(source)}
    if kind == "code":
        result["execution_count"] = None
        result["outputs"] = []
        if hidden:
            result["metadata"] = {"collapsed": True, "jupyter": {"source_hidden": True}}
    return result


def introduction(number: str) -> dict:
    purpose = PURPOSES[number]
    return cell(
        "markdown",
        f"{GUIDE_MARKER}\n"
        "### 실행 안내\n\n"
        f"- **이 노트북의 목적:** {purpose}\n"
        "- **실행 순서:** Colab에서 첫 셀부터 아래로 차례대로 실행합니다. "
        "중간 셀만 단독 실행하면 앞에서 만든 설정과 코드가 없어 오류가 날 수 있습니다.\n"
        "- **수정할 곳:** 앞부분의 경로·실험 설정 셀을 먼저 확인합니다. "
        "내장 실행 코드 셀은 펼쳐서 읽을 수 있지만 보통 수정하지 않습니다.\n"
        "- **결과 확인:** 끝부분의 저장 경로와 보고서 출력에서 결과를 확인합니다. "
        "ZIP과 학습 결과를 혼동하지 않도록 각 단계의 설명을 읽어 주세요.\n",
    )


def embedded_assignment(code: str):
    """Return a one-line SOURCES/SOURCE literal assignment, if one exists."""
    if "SOURCES = " not in code[:250] and "SOURCE = " not in code[:250]:
        return None
    tree = ast.parse(code)
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name) or target.id not in {"SOURCES", "SOURCE"}:
            continue
        if node.lineno != node.end_lineno:
            raise ValueError("Embedded source assignment must occupy one line")
        value = ast.literal_eval(node.value)
        if target.id == "SOURCE":
            if not isinstance(value, str):
                raise ValueError("SOURCE must contain Python source text")
            modules = {"hair_six_class_package": value}
        else:
            if not isinstance(value, dict) or not all(
                isinstance(k, str) and isinstance(v, str) for k, v in value.items()
            ):
                raise ValueError("SOURCES must be a dictionary of source strings")
            modules = value
        return target.id, node.lineno - 1, modules
    return None


def split_embedded_cell(original: dict, stem: str) -> tuple[list[dict], dict[str, str]]:
    code = source_text(original)
    found = embedded_assignment(code)
    if found is None:
        return [original], {}
    variable, line_index, modules = found
    if any(not name.isidentifier() for name in modules):
        raise ValueError(f"Invalid embedded module name in {stem}")
    directory = f"/content/mediflow_notebook_sources/{stem}"
    prefix = [
        cell(
            "code",
            "from pathlib import Path\n"
            f"_EMBEDDED_SOURCE_DIR = Path({directory!r})\n"
            "_EMBEDDED_SOURCE_DIR.mkdir(parents=True, exist_ok=True)\n",
        )
    ]
    expected = {}
    for name, content in modules.items():
        compile(content, name + ".py", "exec")
        expected[name] = hashlib.sha256(content.encode("utf-8")).hexdigest()
        prefix.append(
            cell(
                "markdown",
                f"{EMBEDDED_MARKER}\n"
                f"### 내장 실행 코드: `{name}.py`\n\n"
                "이 셀의 코드는 줄 단위로 읽을 수 있도록 펼쳐 놓았습니다. "
                "Colab 임시 공간에 기록한 뒤 아래 실행 셀에서 원본과 동일한 "
                "SHA-256을 확인합니다. 설정 변경은 앞쪽 설정 셀에서 하세요.\n",
            )
        )
        prefix.append(
            cell(
                "code", f"%%writefile {directory}/{name}.py\n{content}", hidden=True
            )
        )

    if variable == "SOURCES":
        replacement = "SOURCES = {\n" + "".join(
            f"    {name!r}: (_EMBEDDED_SOURCE_DIR / {name + '.py'!r})"
            ".read_text(encoding='utf-8'),\n"
            for name in modules
        ) + "}\n"
        items = "SOURCES"
    else:
        replacement = (
            "SOURCE = (_EMBEDDED_SOURCE_DIR / 'hair_six_class_package.py')"
            ".read_text(encoding='utf-8')\n"
        )
        items = "{'hair_six_class_package': SOURCE}"
    replacement += (
        "import hashlib as _source_hashlib\n"
        f"_EXPECTED_SOURCE_SHA256 = {expected!r}\n"
        f"for _name, _code in {items}.items():\n"
        "    if _source_hashlib.sha256(_code.encode('utf-8')).hexdigest() != "
        "_EXPECTED_SOURCE_SHA256[_name]:\n"
        "        raise ValueError(f'내장 코드가 원본과 다릅니다: {_name}. 셀을 다시 실행하세요.')\n"
    )
    lines = code.splitlines(keepends=True)
    lines[line_index] = replacement
    loader = cell("code", "".join(lines))
    compile(source_text(loader), stem + "_loader.py", "exec")
    return [*prefix, loader], expected


def polish_special_notebook(notebook: dict, number: str) -> bool:
    if number not in EXTRA_SECTIONS or any(
        SECTION_MARKER in source_text(item) for item in notebook["cells"]
    ):
        return False
    updated = []
    headings = EXTRA_RENAMES.get(number, {})
    for original in notebook["cells"]:
        source = source_text(original)
        if original["cell_type"] == "markdown":
            for before, after in headings.items():
                if source.startswith(before):
                    original["source"] = source_lines(after + source[len(before) :])
                    break
        elif original["cell_type"] == "code":
            for prefix, title in EXTRA_SECTIONS[number]:
                if source.startswith(prefix):
                    updated.append(cell("markdown", f"{SECTION_MARKER}\n{title}\n"))
                    break
            if len(source) > 5_000:
                original["metadata"].setdefault("collapsed", True)
                original["metadata"].setdefault("jupyter", {})["source_hidden"] = True
        updated.append(original)
    notebook["cells"] = updated
    return True


def format_notebook(path: Path) -> tuple[bool, dict[str, str]]:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    number = path.name[:2]
    if number not in PURPOSES:
        raise ValueError(f"Unexpected notebook: {path.name}")
    original_cells = notebook["cells"]
    has_guide = any(GUIDE_MARKER in source_text(c) for c in original_cells)
    new_cells = []
    source_hashes = {}
    if not has_guide:
        for index, original in enumerate(original_cells):
            if index == 1:
                new_cells.append(introduction(number))
            if original["cell_type"] == "code":
                replacement, hashes = split_embedded_cell(original, path.stem)
                if hashes:
                    source_hashes.update(hashes)
                    new_cells.extend(replacement)
                    continue
            new_cells.append(original)
        if len(original_cells) == 1:
            new_cells.append(introduction(number))
        notebook["cells"] = new_cells
    polished = polish_special_notebook(notebook, number)
    if has_guide and not polished:
        return False, {}
    new_cells = notebook["cells"]
    used = {c["id"] for c in new_cells if c.get("id")}
    for index, item in enumerate(new_cells):
        if "id" not in item:
            candidate = f"mf-{index:03d}"
            while candidate in used:
                candidate += "x"
            item["id"] = candidate
            used.add(candidate)
    path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return True, source_hashes


def main() -> None:
    changed = 0
    modules = 0
    for path in sorted(NOTEBOOK_DIR.glob("*.ipynb")):
        modified, hashes = format_notebook(path)
        changed += modified
        modules += len(hashes)
        if modified:
            print(f"{path.name}: formatted, {len(hashes)} embedded modules verified")
    print(f"Formatted notebooks: {changed}; embedded modules: {modules}")


if __name__ == "__main__":
    main()
