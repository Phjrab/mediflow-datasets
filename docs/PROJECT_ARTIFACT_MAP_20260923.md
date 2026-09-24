# MediFlow 문서·코드·모델 위치 지도

기준일: 2026-09-23. 파일을 이동하거나 삭제하지 않고 현재 사용할 것과 과거 기록을 구분한다.

## 가장 먼저 볼 파일

| 목적 | 기준 파일 |
|---|---|
| 프로젝트 전체 설명 | `research/MEDIFLOW_PROJECT_COMPREHENSIVE_FINAL_20260923.md` |
| 현재 단계와 남은 일 | `ROADMAP.md` |
| 세 최종 후보 경로·해시·성능 | `../results/CANDIDATE_INDEX.json` |
| v1·v2 학습 방법·성능·선정 이유 비교 | `../results/MODEL_VERSION_COMPARISON.md` |
| 팀 모델 사용법 | `guides/TEAM_MODEL_QUICKSTART.md`, `../results/MODEL_USAGE.md` |
| 노트북 사용법 | `../notebooks/사용안내.md` |

## 최종 후보 모델

| 도메인 | 현재 후보 폴더 | 이전 후보 처리 |
|---|---|---|
| Skin | `results/skin/selected_models/v1/` | 실제 선정 후보는 v1 하나이며 Original/Augmented는 비교 실험 |
| Web Skin | `results/web_skin/selected_models/v2/` | v1은 논문 강화 전 최종 선정 모델, v2는 강화 후 현재 모델 |
| Hair | `results/hair/selected_models/v2/` | v1은 논문 강화 전 최종 선정 모델, v2는 강화 후 현재 모델 |

각 버전의 모델과 선정 이유는 `selected_models/vN`에 있고, 원본 ZIP과 `.zip.sha256`은
`candidates`에 있다. 모델을 선택할 때 폴더 이름만 추측하지 말고
`results/CANDIDATE_INDEX.json`을 읽는다.

## 노트북

| 범위 | 파일 | 상태 |
|---|---|---|
| 공통 | `01_common_dataset_audit_colab.ipynb` | 재사용 가능 |
| 공통 | `02_common_original_vs_augmented_colab.ipynb` | 재사용 가능 |
| 공통 | `03_common_six_experiments_colab.ipynb` | 재사용 가능 |
| Hair | `04`~`09` | 완료된 실험·구현 기록 |
| Hair | `10_hair_b1_384_final_test_package_colab.ipynb` | 실행 완료 |
| Web Skin | `11_web_skin_wsdan_attention_colab.ipynb` | 실행 완료 |
| Web Skin | `12_web_skin_pmg_b1_384_colab.ipynb` | 실행 완료 |
| Web Skin | `13_web_skin_pmg_b0_256_final_test_package_colab.ipynb` | 실행 완료 |
| Web Skin | `14_web_skin_medsiglip_linear_probe_colab.ipynb` | 실행 완료, PMG v2보다 낮아 미채택 |
| Hair | `15_hair_medsiglip_linear_probe_colab.ipynb` | 실행 완료, Hair v2보다 낮아 미채택 |

`common_colab_notebooks_v1.zip`과 `legacy_notebooks_20260909.zip`은 보관용이다.

## 코드

- `src/mediflow_datasets/common_audit.py`: 데이터 기계 검사
- `src/mediflow_datasets/common_engine.py`: 공통 학습·평가 도구
- `src/mediflow_datasets/common_workflow.py`: Colab 실행·기록·재개
- `src/mediflow_datasets/candidate_reproduction.py`: 세 현재 후보 공통 추론 검사
- `src/mediflow_datasets/hair_*.py`: Hair 논문 실험과 최종화
- `src/mediflow_datasets/web_skin_*.py`: Web Skin 논문 실험과 최종화
- `scripts/build_*.py`: 독립 실행 가능한 Colab 노트북 생성기

## 결과 폴더의 의미

- `results/<domain>/selected_models/`: 실제 선정 모델의 v1·v2 사용 파일과 선정 이유
- `results/<domain>/candidates/`: 원본 후보 ZIP과 전체 재현 보고서
- `results/<domain>/experiments/`: 학습 지표·예측·그래프·실험 소스
- `results/<domain>/presentation_*`: 발표용 시각자료
- `results/<domain>/archives/`: 원본 보고서 ZIP 등 보관 자료
- `docs/research/`: 연구 이유와 결과 해석
- `docs/archive/`: 당시 계획, 중간 분석과 검증 과정

기존 결과를 현재 후보로 오해하지 않도록 최종 후보는 `CANDIDATE_INDEX.json`, 현재 해석은
최종 종합 문서를 우선한다. 과거 파일은 재현성과 의사결정 추적을 위해 보존한다.
