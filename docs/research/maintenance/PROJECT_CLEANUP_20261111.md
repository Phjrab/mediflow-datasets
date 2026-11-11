# 4차 학습 전 프로젝트 폴더 정리 · 2026-11-11

> 2026-11-11 정리 상태: **분석·검토 또는 과정 기록, 후속 학습 상태는 최신 로드맵 참조**. [현재 작업 안내](../../README.md) · [전체 문서 대조표](../../DOCUMENT_INDEX.md). 아래 본문은 작성 시점 기록이며, 본문·원본 JSON의 이전 경로는 정리 대조표로 추적한다.

## 현재 단계와 범위

로드맵의 실제 카메라 환경 적응 단계다. 1·2·3차 학습과 최종 Source Test는 완료했고, 4차 촬영 변환 수정본은 학습 직전 검사까지 마친 상태다. 이번 작업은 로컬 폴더 정리이며 새 학습·Drive 변경·새 성능 측정은 수행하지 않았다.

## 삭제한 재생성 파일

| 대상 | 폴더 수 | 파일 수 | 파일 크기 합계 |
| --- | ---: | ---: | ---: |
| `.tmp` 바로 아래의 과거 pytest 검사 임시폴더 | 62 | 32,613 | 5,261,187,422 bytes |
| pytest·ruff 캐시와 코드의 `__pycache__` | 5 | 126 | 1,914,497 bytes |
| 합계 | 67 | 32,739 | 5,263,101,919 bytes |

약 5.26GB의 파일 크기를 정리했다. 이 값은 삭제한 파일의 논리적 크기 합계다. 모든 삭제 경로가 작업 폴더 안에 있는지 확인하고, pytest 폴더는 최상위에 검사 산출물만 있는지 검사했다. 심볼릭 링크·정션과 활성 검사 잠금 여부를 확인했으며 건너뛴 대상은 없었다.

`.tmp` 전체를 삭제하지 않았다. 현재 로컬 그래프 검사에 쓰는 `.tmp/final-test-report-libraries`, 모델 저장 검사의 TEMP 경로인 `.tmp/strength-native-temp`, 보존 여부를 확정하지 않은 기타 작업 폴더는 유지했다. `.venv`와 `.git`도 유지했다.

## 정리한 위치

```text
notebooks/camera_adaptation/
  README.md
  MediFlow_Camera_Revised_Adaptation_20261111.ipynb  ← 현재 4차
  completed/                                      ← 완료 실험 5개
  archive/                                        ← 이전 준비·중단 이력 6개

camera_capture/_review/
  stage1_release_v3_20261110/                      ← 현재 공통 ZIP, 그대로
  training_preparation_archive_20261110/
    stage1_releases/v1/                            ← 이전 공통 ZIP v1
    stage1_releases/v2/                            ← 이전 공통 ZIP v2
  project_cleanup_20261111/                        ← 이번 정리 계획·백업·검증
```

[현재 노트북 안내](../../../notebooks/camera_adaptation/README.md), [완료 실험 목록](../../../notebooks/camera_adaptation/completed/README.md), [보관용 목록](../../../notebooks/camera_adaptation/archive/README.md)에 각 파일의 역할과 실행 상태를 적었다. 노트북 11개는 파일명·내용을 유지하고 하위 폴더만 바꾸었다.

이전 `stage1_release_20261110`과 `stage1_release_v2_20261110`은 각각 위의 `stage1_releases/v1`, `v2`로 이동했다. ZIP과 `package_info.json` 4개 파일은 재압축·수정 없이 보존했다. 현재 공통 ZIP v3의 위치와 내용은 바꾸지 않았다.

## 안내와 생성 경로

- 루트 README, 문서 안내, 노트북 안내, 결과 안내를 연결했다.
- 연구 문서에서는 이동한 노트북과 이전 ZIP의 경로만 갱신하고 수치는 바꾸지 않았다. 변경 전 문서는 로컬 백업에 남겼다.
- [보고서 그림 안내](../../../presentation/report_assets/README.md)를 추가했다. 그림 자체는 이동하거나 수정하지 않았다.
- `presentation/1차`~`4차`는 9월 공개 데이터 연구 구간이며 11월 카메라 적응 단계 번호와 다르다는 설명을 추가했다.
- 기존 발표 안내가 연결하던 `Colab_정리본`, `Colab_원본그래프_모음`, `Colab_스타일_보완그래프` 세 폴더는 정리 전부터 없었다. 이번 작업에서 삭제한 것으로 오해하지 않도록 확인 상태를 적고 없는 폴더 링크를 제외했다.
- 이전 확대 후보 생성기 `scripts/build_adaptation_strength.py`의 고정 출력 경로만 `archive`로 맞췄다. 현재 4차 생성기·학습 코드·설정은 보존했다. 다른 단계의 생성기는 `--notebook`으로 출력 경로를 받으므로 변경하지 않았다.

`output`의 실행한 분석 3폴더와 `pdf`, 원본 촬영 사진, `data_examples`, 기존·적응 모델, 결과 JSON/CSV, 현재 분석 입력의 경로는 유지했다. 모델이 중복으로 보이는 `presentation/models`와 `results/selected_models` 등의 사본도 사용 계약과 연결 경로를 위해 유지했다.

## 확인 결과

원본·모델·수치·설정 등 보호 목록 781개와 현재 런타임·생성기·공통 ZIP 7개를 이동 전 SHA-256(파일 내용의 식별값)과 대조했다. 모두 일치했다. 이동한 노트북 11개와 이전 ZIP 관련 4개 파일도 원본과 일치했다. 실제 촬영 사진은 124장으로 유지됐다.

현재 4차 노트북·설정 파일의 내용과 위치는 그대로다. 설정 SHA-256은 `492ee01afb4579b6402fca29459c59d56618243589642fc022db18bbbe45a165`이며, 생성기로 메모리에서 재구성한 노트북도 현재 저장본과 일치했다.

`ruff check --no-cache src tests scripts/build_adaptation_strength.py`를 통과했다. 전체 pytest는 **191개 통과·3개 건너뜀**으로 종료했다. 건너뛴 3개는 기본 가상환경에 `matplotlib`이 없어 생략된 기존 그래프 검사다. 이번 변경은 학습 과정·변환·추론 코드를 바꾸지 않았다. 검사에서 새로 만든 임시파일 1,219개도 종료 후 별도로 정리했다.

갱신한 Markdown 안내·보고서 24개의 로컬 링크 280개를 확인했고 연결되지 않는 링크는 없었다. `git diff --check`도 통과했다.

## 재현 근거

로컬 전용 [정리 증거 폴더](../../../camera_capture/_review/history/project_cleanup)에 다음을 남긴다.

- `cleanup_plan.json`, `path_map.json`: 삭제·이동 전 경로와 원본 파일 해시.
- `operation_journal.jsonl`, `execution_summary.json`: 실제 삭제·이동 기록.
- `before_cleanup/`, `navigation_changes.json`: 안내 수정 전 사본과 변경 목록.
- `protected_runtime.json`, `verification.json`: 현재 실행 입력과 보존 파일의 대조.
- `pytest.log`: 이번 전체 검사 출력.

이번 정리의 기준 Git commit은 `08cb827aa3c6858e61c71faf7f6481f3a139530b`이다. 정리 변경은 아직 commit·push하지 않았다. 학습 설정·클래스 순서·전처리·학습량은 바뀌지 않았으며, 새 실험 결과가 생긴 것은 아니다.

## 다음 단계와 사용자가 할 일

다음 작업은 [현재 4차 노트북](../../../notebooks/camera_adaptation/completed/MediFlow_Adaptation_Stage4_20261111.ipynb)의 Colab 실행과 Validation 비교다. 이번 정리로 Drive 파일은 변경하지 않았고 기존 공통 ZIP v3를 그대로 쓴다. 폴더 정리를 위해 사용자가 추가로 할 일은 없다.
