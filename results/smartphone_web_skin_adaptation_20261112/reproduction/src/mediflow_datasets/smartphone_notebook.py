"""Generate a Run-all Colab notebook with a pinned private smartphone runtime."""

# ruff: noqa: E501

from __future__ import annotations

import textwrap


def build_notebook(
    bundle_sha256,
    *,
    stage2_reused=True,
    stage2_profile=None,
    stage4_profile=None,
    appearance_requested=False,
):
    def cell(kind, source, identifier):
        value = {
            "cell_type": kind,
            "id": identifier,
            "metadata": {},
            "source": textwrap.dedent(source).strip().splitlines(keepends=True),
        }
        if kind == "code":
            value.update(execution_count=None, outputs=[])
        return value

    book = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "colab": {"provenance": []},
            "accelerator": "GPU",
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        },
        "cells": [
            cell(
                "markdown",
                """
                # MediFlow 스마트폰 얼굴 적응 · 1~4차 한 번에 실행

                **GPU 선택 → 전용 ZIP을 MyDrive/mediflow_Project에 업로드 → 모두 실행**.
                기존 Drive의 `mediflow_Project/datasets`에 있는 web_skin 원본 ZIP을 재사용합니다.
                휴대폰 사진 원본을 다시 올릴 필요가 없습니다. 이 ZIP에는 암호화한 모델 입력이 있습니다.
                복호화 키는 함께 만든 `MediFlow_Smartphone_Target_Key_20261111.txt` 한 줄을
                가려진 입력란에 한 번 붙여넣습니다. 키를 코드 셀에 쓰거나 노트북에 저장하지 않습니다.

                - web_skin만: 건선·아토피·여드름·정상·주사. 피부/두피는 실행하지 않습니다.
                - 1차: 기존 초기 B0·224와 선정 PMG·B0·256의 스마트폰 조명 적응 각 5 epoch.
                - 2차: 1차에서 고른 모델의 학습 전 저장본 + 측정 근거가 있는 색·미세 대비·흐림.
                - 3차: 같은 출발점·2차 변환 + MMD와 CORAL 학습 손실, 각 5 epoch.
                - 4차: 같은 출발점 + 스마트폰 Train 표본 재생으로 재점검한 촬영 변환, 5 epoch.
                - 총 5회이며 GPU 하나에서 순서대로 실행합니다. 단계 결과를 이어 학습하지 않습니다.
                - 같은 분할·seed 42·Adam 1e-5·변환 확률 0.5. Validation으로만 모델을 고릅니다.
                - 스마트폰 입력은 EXIF 회전과 ICC→sRGB 변환 후 중앙 정사각형으로 자릅니다.

                5명 중 P01~P04 42장은 설정·정렬용, P05 10장은 최종 선정 후 예측 확인용입니다.
                질환 정답이 없어 Target Accuracy는 계산하지 않습니다. 이번에는 Source Test도 평가하지 않습니다.
                기존 공통 ZIP·웹캠 결과와 별도 결과 폴더를 사용합니다.
                중단되면 같은 노트북을 다시 모두 실행하고 같은 키를 입력합니다.
                완료 실험을 건너뛰고 마지막 저장 epoch에서 재개합니다.
            """,
                "guide",
            ),
            cell(
                "code",
                f'''
                from google.colab import drive
                from pathlib import Path
                import hashlib, json, os, subprocess, sys, zipfile
                drive.mount("/content/drive")
                PROJECT = Path("/content/drive/MyDrive/mediflow_Project")
                ZIP_PATH = PROJECT / "MediFlow_Smartphone_WebSkin_Runtime_20261111.zip"
                EXPECTED_SHA256 = "{bundle_sha256}"
                if not ZIP_PATH.is_file():
                    raise FileNotFoundError(f"전용 실행 ZIP을 이 위치에 올려주세요: {{ZIP_PATH}}")
                def digest(path):
                    h = hashlib.sha256()
                    with path.open("rb") as f:
                        for block in iter(lambda: f.read(1024*1024), b""): h.update(block)
                    return h.hexdigest()
                print("스마트폰 ZIP 확인 중", flush=True)
                if digest(ZIP_PATH) != EXPECTED_SHA256:
                    raise ValueError("전용 ZIP이 이 노트북의 고정 파일과 다릅니다.")
                RUNTIME = Path("/content") / ("mediflow_phone_"+EXPECTED_SHA256[:12])
                if not (RUNTIME/".runtime_ready").is_file():
                    RUNTIME.mkdir(exist_ok=True)
                    from pathlib import PurePosixPath
                    with zipfile.ZipFile(ZIP_PATH) as archive:
                        names = archive.namelist()
                        if len(names) != len(set(n.casefold() for n in names)):
                            raise ValueError("중복 ZIP 경로")
                        for name in names:
                            p = PurePosixPath(name)
                            if p.is_absolute() or ".." in p.parts or "\\\\" in name or ":" in name:
                                raise ValueError("안전하지 않은 ZIP 경로")
                        if archive.testzip(): raise ValueError("ZIP 손상")
                        archive.extractall(RUNTIME)
                manifest = json.loads((RUNTIME/"runtime_manifest.json").read_text())
                for name, sha in manifest["files"].items():
                    if digest(RUNTIME/name) != sha: raise ValueError("실행 파일 해시 불일치: "+name)
                (RUNTIME/".runtime_ready").write_text(EXPECTED_SHA256)
                OUTPUT = Path("/content/drive/MyDrive/MediFlow_smartphone_runs") / EXPECTED_SHA256[:12]
                OUTPUT.mkdir(parents=True, exist_ok=True)
                LOCAL = Path("/content") / ("mediflow_phone_work_"+EXPECTED_SHA256[:12])
                LOCAL.mkdir(exist_ok=True)
                print("저장 위치:", OUTPUT)
            ''',
                "drive-and-runtime",
            ),
            cell(
                "code",
                """
                # Colab 기본 TF 패키지를 바꾸지 않는 별도 실행 환경입니다.
                VENV = Path("/content/mediflow_phone_env")
                PYTHON = VENV/"bin/python"
                requirements = ["tensorflow[and-cuda]==2.20.0", "keras==3.13.2", "numpy>=1.26,<3",
                                "pillow>=10,<13", "opencv-python-headless>=4.10,<5",
                                "matplotlib>=3.8,<4", "cryptography>=44,<49", "tqdm>=4.66,<5"]
                probe = "import importlib.metadata as m; assert m.version('tensorflow')=='2.20.0'; assert m.version('keras')=='3.13.2'; import cryptography,cv2,matplotlib,PIL,tqdm"
                gpu = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True)
                if gpu.returncode or not gpu.stdout.strip():
                    raise RuntimeError("Colab 런타임에서 GPU를 선택한 뒤 다시 실행하세요.")
                print("GPU:",gpu.stdout.strip())
                ready = PYTHON.exists() and subprocess.run([str(PYTHON), "-c", probe], capture_output=True).returncode == 0
                if not ready:
                    pip_ready = PYTHON.exists() and subprocess.run(
                        [str(PYTHON), "-m", "pip", "--version"], capture_output=True).returncode == 0
                    if not pip_ready:
                        created = subprocess.run([sys.executable, "-m", "venv", str(VENV)], capture_output=True, text=True)
                        if created.returncode:
                            # Some Colab images omit ensurepip. Bootstrap only the environment creator.
                            print("기본 venv 준비를 보완합니다.",flush=True)
                            subprocess.run([sys.executable,"-m","pip","install","-q","virtualenv>=20,<21"],check=True)
                            subprocess.run([sys.executable,"-m","virtualenv",str(VENV)],check=True)
                    subprocess.run([str(PYTHON), "-m", "pip", "install", *requirements], check=True)
                ENV = {**os.environ, "PYTHONPATH":str(RUNTIME/"src"), "PYTHONUNBUFFERED":"1",
                       "TF_NUM_INTEROP_THREADS":"2", "TF_NUM_INTRAOP_THREADS":"4",
                       "TF_CPP_MIN_LOG_LEVEL":"2", "KERAS_BACKEND":"tensorflow",
                       "MPLBACKEND":"Agg", "PYTHONDONTWRITEBYTECODE":"1"}
                ENV.pop("TF_USE_LEGACY_KERAS", None)
                source_code = "from pathlib import Path; from mediflow_datasets.domain_adaptation_batch import find_source; import json; r=Path("+repr(str(RUNTIME))+"); s=json.loads((r/'study.json').read_text()); print(find_source(Path("+repr(str(PROJECT/"datasets"))+"),s['task']))"
                found = subprocess.run([str(PYTHON), "-c", source_code], env=ENV, capture_output=True, text=True, check=True)
                SOURCE = Path(found.stdout.strip().splitlines()[-1])
                if not SOURCE.is_file(): raise FileNotFoundError("기존 web_skin Source ZIP을 확인하세요.")
                print("기존 Source 데이터:", SOURCE)
            """,
                "isolated-environment",
            ),
            cell(
                "code",
                """
                import getpass, uuid
                from datetime import datetime, timezone
                key = getpass.getpass("스마트폰 전용 복호화 키 한 줄: ").strip()
                if not key: raise ValueError("전용 키 파일 한 줄을 입력하세요.")
                log_path = OUTPUT/("training_"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")+"_"+uuid.uuid4().hex[:6]+".log")
                command = [str(PYTHON), "-m", "mediflow_datasets.smartphone_training",
                           "--runtime",str(RUNTIME), "--source",str(SOURCE),
                           "--output",str(OUTPUT), "--local",str(LOCAL)]
                # A lock prevents duplicate Run-all cells from touching the same checkpoints.
                import fcntl
                with (LOCAL/"execution.lock").open("a") as lock:
                    try: fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
                    except BlockingIOError:
                        del key
                        raise RuntimeError("같은 학습이 이미 실행 중입니다.") from None
                    process = None
                    try:
                        with log_path.open("x",encoding="utf-8") as log:
                            process = subprocess.Popen(command,env=ENV,stdin=subprocess.PIPE,
                                stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
                            with process.stdin: process.stdin.write(key+"\\n")
                            del key
                            for line in process.stdout:
                                print(line,end="",flush=True); log.write(line); log.flush()
                            if process.wait(): raise RuntimeError("미완료: 위 로그를 확인하고 같은 노트북을 다시 실행하세요.")
                    finally:
                        if "key" in globals(): del key
                        if process is not None:
                            if process.poll() is None: process.terminate(); process.wait()
                            process.stdout.close()
                print("실행 로그:",log_path)
            """,
                "run-five-trials",
            ),
            cell(
                "code",
                """
                from IPython.display import display, Image
                summary = json.loads((OUTPUT/"summary.json").read_text())
                if summary.get("status") != "completed": raise RuntimeError("전체 학습이 완료되지 않았습니다.")
                print(json.dumps(summary,ensure_ascii=False,indent=2))
                for p in sorted(OUTPUT.glob("stage*/learning_curves.png")):
                    print(p.parent.name); display(Image(filename=str(p)))
                display(Image(filename=str(OUTPUT/"validation_comparison.png")))
                RESULT_ZIP = OUTPUT/"MediFlow_Smartphone_WebSkin_Results_20261111.zip"
                if not RESULT_ZIP.is_file(): raise FileNotFoundError("결과 ZIP이 없습니다.")
                print("Drive 결과 ZIP:",RESULT_ZIP)
                print("ZIP SHA-256:",digest(RESULT_ZIP))
                from google.colab import files
                files.download(str(RESULT_ZIP))
            """,
                "results-and-download",
            ),
        ],
    }
    if stage2_reused:
        guide = "".join(book["cells"][0]["source"])
        guide = (
            guide.replace(
                "- 2차: 1차에서 고른 모델의 학습 전 저장본 + 측정 근거가 있는 색·미세 대비·흐림.",
                "- 2차: 측정상 색·미세 대비·흐림을 추가할 근거가 부족해 1차와 동일합니다. 선택된 1차 결과를 재사용합니다.",
            )
            .replace(
                "- 총 5회이며 GPU 하나에서 순서대로 실행합니다.",
                "- 실제 학습은 총 4회이며 GPU 하나에서 순서대로 실행합니다.",
            )
            .replace(
                "예측 확인용입니다.",
                "예측 확인용입니다. 2차와 1차, phone_expanded와 phone_lighting은 동일 조건의 별칭입니다.",
            )
        )
        book["cells"][0]["source"] = guide.splitlines(keepends=True)
    elif stage2_profile is not None:
        terms = []
        for key, identity, label in (
            ("color_a", [0.0, 0.0], "Lab a 색 이동"),
            ("color_b", [0.0, 0.0], "Lab b 색 이동"),
            ("texture_gain", [1.0, 1.0], "미세 대비 배율"),
            ("blur_sigma_roi64", [0.0, 0.0], "흐림 sigma(ROI64)"),
        ):
            if stage2_profile[key] != identity:
                terms.append(f"{label} {stage2_profile[key]}")
        applied = "; ".join(terms) if terms else "추가 외형 변환 없음"
        guide = "".join(book["cells"][0]["source"]).replace(
            "- 2차: 1차에서 고른 모델의 학습 전 저장본 + 측정 근거가 있는 색·미세 대비·흐림.",
            "- 2차: 1차에서 고른 모델의 학습 전 저장본 + 1차 조명 + "
            + applied
            + ". 아래 고정 설정에 없는 변환은 추가하지 않습니다. 5 epoch.",
        )
        book["cells"][0]["source"] = guide.splitlines(keepends=True)
    if appearance_requested:
        guide = "".join(book["cells"][0]["source"])
        p2, p4 = stage2_profile, stage4_profile
        weak = p2["blur_sigma_roi64"][1] * p2["blur_input_scale"]
        strong = p4["blur_sigma_roi64"][1] * p4["blur_input_scale"]
        guide = guide.replace(
            "- 4차: 같은 출발점 + 스마트폰 Train 표본 재생으로 재점검한 촬영 변환, 5 epoch.",
            f"- 4차: 같은 출발점 + 기존 4차 조명 + 미세 대비 {p4['texture_gain']} + "
            f"흐림 sigma 0~{strong:g}px(256px 기준), 5 epoch. 색 이동은 추가하지 않습니다.",
        )
        guide += (
            f"\n\n사용자 요청 강도 비교: 2·3차 미세 대비 {p2['texture_gain']}, "
            f"흐림 sigma 0~{weak:g}px; 4차 0~{strong:g}px(256px 입력 기준). "
            "224px 모델이면 흐림 범위도 224/256배입니다. "
            "추가 질감·흐림은 분포 근접성 기준에서 채택된 값이 아니라 탐색 실험입니다. "
            "성능 개선이나 실제 휴대폰 초점과의 일치는 아직 확인하지 않았습니다."
        )
        book["cells"][0]["source"] = guide.splitlines(keepends=True)
    return book
