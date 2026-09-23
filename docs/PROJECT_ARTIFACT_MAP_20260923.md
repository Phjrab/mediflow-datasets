# MediFlow 문서·코드·모델 위치 지도

기준일: 2026-09-23. 파일을 이동하거나 삭제하지 않고 현재 사용할 것과 과거 기록을 구분한다.

## 가장 먼저 볼 파일

| 목적 | 기준 파일 |
|---|---|
| 프로젝트 전체 설명 | `research/MEDIFLOW_PROJECT_COMPREHENSIVE_FINAL_20260923.md` |
| 현재 단계와 남은 일 | `ROADMAP.md` |
| 세 최종 후보 경로·해시·성능 | `../results/CANDIDATE_INDEX.json` |
| 팀 모델 사용법 | `guides/TEAM_MODEL_QUICKSTART.md`, `../results/MODEL_USAGE.md` |
| 노트북 사용법 | `../notebooks/사용안내.md` |

## 최종 후보 모델

| 도메인 | 현재 후보 폴더 | 이전 후보 처리 |
|---|---|---|
| Skin | `results/skin/candidates/public_candidate_v1_b0_224_ce_augmented_20260909_075056/` | 현재 후보 |
| Web Skin | `results/web_skin/candidates/public_candidate_v2_pmg_b0_256_ce_20260922_235840_093d10de/` | v1은 과거 비교 기록으로 보존 |
| Hair | `results/hair/candidates/public_candidate_v2_b1_384_ls005_adam_20260921_155906_82311da4/` | v1은 과거 비교 기록으로 보존 |

각 후보의 원본 ZIP과 `.zip.sha256`은 같은 `candidates` 폴더에 있다. 모델을 선택할 때 폴더
이름만 추측하지 말고 `results/CANDIDATE_INDEX.json`을 읽는다.

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

- `results/<domain>/candidates/`: 현재 통합 후보와 과거 후보
- `results/<domain>/experiments/`: 학습 지표·예측·그래프·실험 소스
- `results/<domain>/presentation_*`: 발표용 시각자료
- `results/<domain>/archives/`: 원본 보고서 ZIP 등 보관 자료
- `docs/research/`: 연구 이유와 결과 해석
- `docs/archive/`: 당시 계획, 중간 분석과 검증 과정

기존 결과를 현재 후보로 오해하지 않도록 최종 후보는 `CANDIDATE_INDEX.json`, 현재 해석은
최종 종합 문서를 우선한다. 과거 파일은 재현성과 의사결정 추적을 위해 보존한다.
