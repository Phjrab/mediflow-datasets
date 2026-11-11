"""Build Stage-4-only Test inference with embedded, verified historical metrics."""

import argparse
import ast
import copy
import hashlib
import json
import subprocess
from pathlib import Path
from zipfile import ZipFile

import build_adaptation_final_test as previous

ROOT = Path(__file__).resolve().parents[1]
STAGE4 = ROOT / "results/camera_adaptation_stage4_before_refit_20261111"
STUDY_SHA = "492ee01afb4579b6402fca29459c59d56618243589642fc022db18bbbe45a165"
RUNTIME = (
    "adaptation_stage4_test.py",
    "adaptation_final_test.py",
    "adaptation_stage1.py",
    "camera_extended.py",
    "domain_adaptation_experiment.py",
)
PREVIOUS_TEST = ROOT / "results/camera_adaptation_final_test_20261110"


def lock_plan():
    plan = copy.deepcopy(previous.read(ROOT / "configs/adaptation_final_test_20261110.json"))
    wrapper = previous.read(STAGE4 / "study.json")
    study = wrapper["study"]
    snapshot = ROOT / (
        "presentation/report_assets/revised_pretraining_check_20261111/"
        "checked_inputs/camera_revised_20261111.json"
    )
    if (
        wrapper["study_sha256"] != STUDY_SHA
        or previous.sha256(snapshot) != STUDY_SHA
        or previous.read(snapshot) != study
        or study["configuration_sha256"] != plan["configuration_sha256"]
        or study["bundle_sha256"] != plan["bundle_sha256"]
        or previous.sha256(previous.BUNDLE) != plan["bundle_sha256"]
    ):
        raise ValueError("Expected the verified, completed pre-refit Stage 4")
    selection_file = STAGE4 / "selection.json"
    selected = previous.read(selection_file)["selected_on_validation"]
    mapping = {
        "initial": "initial",
        "previous_expanded": "stage2_expanded",
        "stronger_expanded": "stage4_revised",
    }
    plan.update(
        schema="mediflow_stage4_only_source_test_v2",
        evaluate_arms=["stage4_revised"],
        input_policies={},
        previous_test={
            "plan_sha256": previous.read(PREVIOUS_TEST / "summary.json")["plan_sha256"],
            "files": {
                name: previous.sha256(PREVIOUS_TEST / name)
                for name in (
                    "evaluation_plan.json",
                    "summary.json",
                    "skin.json",
                    "web_skin.json",
                    "hair.json",
                )
            },
        },
        stage4_study_sha256=STUDY_SHA,
        stage4_study_file_sha256=previous.sha256(STAGE4 / "study.json"),
        stage4_drive_folder="measured_strength_" + STUDY_SHA[:12],
        stage4_selection_sha256=previous.sha256(selection_file),
        stage4_profiles={},
        conditions=["clean", "lighting", "expanded", "stage4_lighting", "stage4_expanded"],
        selection_scope=(
            "Fixed from Stage 4 Validation comparison among initial, stage2 and stage4; "
            "stage1 and stage3 are historical comparison-only, not newly evaluated."
        ),
        selection_rule=previous.read(selection_file)["criteria"],
        prior_source_test_inspected=True,
        test_used_for_model_selection=False,
        refitted_stage4_excluded=True,
        git_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        runtime_sha256={
            name: previous.sha256(ROOT / previous.MODULE_DIR / name) for name in RUNTIME
        },
    )
    if plan["runtime_sha256"]["adaptation_stage1.py"] != plan["predictor_code_sha256"]:
        raise ValueError("Predictor changed since the previous fixed evaluation")
    plan["builder_sha256"] = previous.sha256(Path(__file__))
    verification = previous.read(PREVIOUS_TEST / "import_verification.json")
    for name, sha in plan["previous_test"]["files"].items():
        if sha != verification["imported_members_sha256"][name]:
            raise ValueError(f"Verified previous Test result changed: {name}")
    old_record = previous.read(PREVIOUS_TEST / "evaluation_plan.json")
    if (
        old_record["plan"] != previous.read(ROOT / "configs/adaptation_final_test_20261110.json")
        or old_record["plan_sha256"] != plan["previous_test"]["plan_sha256"]
    ):
        raise ValueError("Previous measured Test does not match its frozen plan")
    with ZipFile(previous.BUNDLE) as bundle:
        config = json.loads(bundle.read("configs/adaptation_stage1.json"))
        if previous.object_hash(config) != plan["configuration_sha256"]:
            raise ValueError("Common bundle configuration changed")
        for domain, lock in plan["domains"].items():
            task = config["tasks"][domain]
            plan["input_policies"][domain] = task["input_policy"]
            spec = task["models"][lock["start_role"]]
            run = previous.read(STAGE4 / domain / "run_configuration.json")
            complete = previous.read(STAGE4 / domain / "complete.json")
            identity = previous.read(STAGE4 / domain / "identity.json")
            if (
                complete["model_config"] != {**spec, "role": lock["start_role"]}
                or run["training_configuration"] != config
                or run["profile"] != study["domains"][domain]["profile"]
                or identity["study_sha256"] != STUDY_SHA
                or identity["source_manifest_sha256"] != lock["manifest_sha256"]
                or identity["training_sha256"] != plan["predictor_code_sha256"]
                or identity["camera_transform_sha256"]
                != plan["runtime_sha256"]["camera_extended.py"]
                or len(previous.read(STAGE4 / domain / "expanded/history.json")) != 5
            ):
                raise ValueError(f"{domain}: completed Stage 4 contract differs")
            lock["models"]["stage4_revised"] = {
                "stage": 4,
                "path": f"{domain}/{complete['best_model']['file']}",
                "sha256": complete["best_model"]["sha256"],
            }
            lock["classes"] = task["classes"]
            lock["source_sha256"] = task["source_sha256"]
            lock["model_spec"] = spec
            lock["selected_on_validation"] = mapping[selected[domain]]
            lock["selection_validation_macro_f1"] = {
                mapping[arm]: {
                    condition: values[condition]["macro_f1"]
                    for condition in ("clean", "weak", "expanded")
                }
                for arm, values in complete["validation"].items()
            }
            lock["stage4_validation_report_sha256"] = previous.sha256(
                STAGE4 / domain / "complete.json"
            )
            roots = (ROOT, *previous.STAGES, STAGE4)
            for info in lock["models"].values():
                if previous.sha256(roots[info["stage"]] / info["path"]) != info["sha256"]:
                    raise ValueError(f"{domain}: checkpoint bytes changed")
            if (
                previous.sha256(previous.STAGES[0] / domain / "source_manifest.json")
                != lock["manifest_sha256"]
            ):
                raise ValueError(f"{domain}: Test manifest changed")
            plan["stage4_profiles"][domain] = study["domains"][domain]["profile"]
    return plan


def notebook(plan_json):
    plan = json.loads(plan_json)
    book = previous.notebook(plan_json)
    source = "".join(book["cells"][1]["source"])
    prior_files = {
        name: (PREVIOUS_TEST / name).read_bytes()
        for name in plan.get("previous_test", {}).get("files", {})
    }
    prior_code = (
        '    prior = runtime / "previous_test"\n'
        "    prior.mkdir()\n"
        f"    for name, body in {prior_files!r}.items():\n"
        "        (prior / name).write_bytes(body)\n"
    )
    replacements = {
        'stages = [run_root / f"stage{n}" for n in (1, 2, 3)]': (
            'stages = [run_root / f"stage{n}" for n in (1, 2, 3)]\n'
            f"stages.append(run_root / {plan['stage4_drive_folder']!r})"
        ),
        "for stage in stages):": (
            'for stage in stages[:1]) or not (stages[3] / "study.json").is_file():'
        ),
        "Drive에서 1·2·3차 완료 폴더를 확인하세요.": (
            "Drive에서 1차 분할 목록과 조정 전 4차 완료 폴더를 확인하세요."
        ),
        'out = run_root / "final_test_': 'out = run_root / "stage4_test_',
        '    plan_path = work / "final_test_plan.json"': (
            '    (runtime / "src/mediflow_datasets/adaptation_stage4_test.py").write_bytes('
            f"{(ROOT / previous.MODULE_DIR / 'adaptation_stage4_test.py').read_bytes()!r})\n"
            + prior_code
            + '    plan_path = work / "final_test_plan.json"'
        ),
        "mediflow_datasets.adaptation_final_test": "mediflow_datasets.adaptation_stage4_test",
        "MediFlow_Final_Test_Results_20261110.zip": "MediFlow_Stage4_Test_Results_20261111.zip",
        'display(Image(filename=str(out / "plots/selected_confusion_matrices.png")))': (
            'display(Image(filename=str(out / "plots/stage4_confusion_matrices.png")))'
        ),
        "MediFlow 최종 Test · 그래프 출력 설정 자동 처리 수정본": (
            "MediFlow 4차만 Test · 코드 사본 검증 수정본"
        ),
    }
    for before, after in replacements.items():
        if before not in source:
            raise ValueError(f"Colab bootstrap changed: {before}")
        source = source.replace(before, after)
    env_line = '    env = {**os.environ, "PYTHONPATH":'
    if source.count(env_line) != 1:
        raise ValueError("Expected one notebook process environment")
    source = source.replace(
        env_line,
        f"    for name, expected in {plan.get('runtime_sha256', {})!r}.items():\n"
        '        staged = runtime / "src/mediflow_datasets" / name\n'
        "        if hashlib.sha256(staged.read_bytes()).hexdigest() != expected:\n"
        '            raise ValueError(f"평가 코드 사본 검사 실패: {name}")\n'
        '    print("평가 코드 사본 검사 완료", flush=True)\n' + env_line,
    )
    payloads = {
        node.value
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Constant) and isinstance(node.value, bytes)
    }
    for name in ("adaptation_stage4_test.py", "adaptation_final_test.py", "adaptation_stage1.py"):
        if (ROOT / previous.MODULE_DIR / name).read_bytes() not in payloads:
            raise ValueError(f"Notebook runtime payload was modified while packaging: {name}")
    compile(source, "stage4-test-colab", "exec")
    book["cells"][1]["source"] = source.splitlines(True)
    output_name = "stage4_test_" + hashlib.sha256(plan_json.encode()).hexdigest()[:12]
    guide = f"""# MediFlow 4차만 추가 Test · 기본~3차 결과 재사용

**GPU 선택 → 모두 실행 → Google Drive 연결** 순서로 실행합니다.
새 ZIP 업로드, Target 복호화 키 입력, 추가 학습은 필요 없습니다.
기존 공통 ZIP v3·원본 데이터 ZIP·1차 분할 목록과 **조정 전 4차** 저장 모델을 사용합니다.
기본·1·2·3차의 검증된 Test 결과는 이 노트북에 포함했으며 다시 추론하지 않습니다.
재설정 4차는 사용자 결정으로 실행하지 않습니다.

피부 700장·얼굴 400장·두피 1,252장의 원본 Test를 보존합니다.
4차 모델 3개에 원본·기존 조명·기존 확장·4차 조명·4차 확장을 적용해 **15개 조합만 새로 평가**합니다.
기본~3차 기존 36개 결과를 재사용해 기존 세 조건에서 비교합니다.
새 4차 변환 두 조건은 4차만 측정하므로 이전 모델 대비 변화량을 비워 둡니다.
난수 seed 1707, 클래스 순서, 입력 크기, float32 RGB 0–255를 유지합니다.
이전에 확인한 Test를 다시 쓰는 후속 비교이며 새로운 미관측 자료의 검증으로 표현하지 않습니다.

후속 후보는 4차 Validation 비교에서 피부·얼굴 4차, 두피 2차로 고정했습니다.
이 선정은 기본·2차·4차 범위이며 1차·3차는 이전 Test 기록으로 비교합니다.
Test 점수로 후보나 변환을 다시 정하지 않습니다.
실제 장비 질환 정확도는 정답 자료가 없어 측정하지 않습니다.

Drive 결과: `MediFlow_adaptation_runs/{plan["bundle_sha256"][:16]}/{output_name}`

진행 장수·실제 오류·로그를 표시하고 완료 분야는 같은 계획으로 재사용합니다.
비교표·클래스별 지표·혼동행렬·비교 그래프·사진별 모델 점수를 저장합니다.
완료하면 `MediFlow_Stage4_Test_Results_20261111.zip`을 다운로드합니다.
모델과 원본 이미지는 결과 ZIP에 중복 포함하지 않습니다.
"""
    book["cells"][0]["source"] = guide.splitlines(True)
    return book


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--notebook", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    plan_json = json.dumps(lock_plan(), ensure_ascii=False, indent=2) + "\n"
    book = notebook(plan_json)
    if not args.overwrite and (args.plan.exists() or args.notebook.exists()):
        raise FileExistsError("Existing prepared plan/notebook; pass --overwrite explicitly")
    for path, contents in (
        (args.plan, plan_json),
        (args.notebook, json.dumps(book, ensure_ascii=False, indent=1) + "\n"),
    ):
        with path.open("w" if args.overwrite else "x", encoding="utf-8", newline="\n") as handle:
            handle.write(contents)
    print(args.notebook)


if __name__ == "__main__":
    main()
