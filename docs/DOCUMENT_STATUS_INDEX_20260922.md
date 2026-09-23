# MediFlow 문서·노트북 상태 색인

기준일: 2026-09-23. 파일을 이동하거나 삭제하지 않고, 현재 기준과 과거 기록의 역할을 구분한다.

## 1. 현재 기준 문서

| 우선순위 | 문서 | 용도 |
|---:|---|---|
| 1 | `research/MEDIFLOW_PROJECT_COMPREHENSIVE_FINAL_20260923.md` | 프로젝트 전체 과정·논문 근거·최종 결과 |
| 2 | `PROJECT_ARTIFACT_MAP_20260923.md` | 문서·코드·모델 위치와 현재/과거 구분 |
| 3 | `ROADMAP.md` | 현재 단계와 다음 실행 순서 |
| 4 | `../results/CANDIDATE_INDEX.json` | 최종 후보 ID·경로·해시·성능의 기계 판독 기준 |
| 5 | `guides/TEAM_MODEL_QUICKSTART.md` | 팀원용 모델 사용법 |
| 6 | `research/PAPER_EXPERIMENTS_FINAL_ANALYSIS_20260922.md` | Hair 논문 기반 방법의 상세 결과 |

## 2. 도메인별 상세 근거

| 도메인 | 문서 | 상태 |
|---|---|---|
| Hair | `research/HAIR_FINAL_CANDIDATE_V2_20260921.md` | 현재 최종 후보 상세 |
| Hair | `research/HAIR_EXPERIMENT_SUMMARY_20260907.md` | 초기 clean·6개 실험 기록 |
| Hair | `research/HAIR_SUPCON_RESULT_20260917.md` | SupCon 결과 근거 |
| Hair | `research/HAIR_REMAINING_METHODS_RESULT_20260917.md` | DINOv2·EfficientNetV2-S·B1 384 근거 |
| Hair | `research/HAIR_CANDIDATE_COMPLEMENTARITY_20260921.md` | 앙상블 검토 근거 |
| Hair | `research/HAIR_SAM_RESULT_20260921.md` | SAM 비교 근거 |
| Web Skin | `research/WEB_SKIN_FULL_SUMMARY_20260908.md` | 데이터·6개 실험·후보 상세 |
| Web Skin | `research/WEB_SKIN_WSDAN_EXPERIMENT_PLAN_20260922.md` | WS-DAN·PMG·MixStyle 실험 설계 |
| Web Skin | `research/WEB_SKIN_PAPER_METHOD_RESULT_20260922.md` | 세 방법 Validation 결과와 PMG 후보 고정 근거 |
| Web Skin | `research/WEB_SKIN_PMG_B1_384_EXPERIMENT_PLAN_20260922.md` | PMG·B1·384 마지막 Validation 실험 설계 |
| Web Skin | `research/WEB_SKIN_PMG_B1_384_RESULT_20260923.md` | PMG·B1·384 Validation 성능 선두 결과 |
| Web Skin | `research/WEB_SKIN_PMG_FINAL_SELECTION_20260923.md` | B0·256 최종 Test 후보의 성능·비용 판단과 고정 계약 |
| Web Skin | `research/WEB_SKIN_FINAL_CANDIDATE_V2_20260923.md` | PMG·B0·256 최종 Test, 패키지와 추론 계약 |
| Skin | `research/SKIN_DATA_AUDIT_20260909.md` | 중복 검사와 clean 구성 |
| Skin | `research/SKIN_ORIGINAL_VS_AUGMENTED_20260909.md` | Original/Augmented 비교 |
| Skin | `research/SKIN_FULL_SUMMARY_20260909.md` | 최종 후보 상세 |

## 3. 배경·설명·발표 자료

| 문서 | 용도 |
|---|---|
| `research/PROJECT_BACKGROUND.md` | 세 전문 모델을 분리한 연구 배경 |
| `research/COMMON_COLAB_NOTEBOOKS_GUIDE.md` | 공통 Colab 사용과 결과 저장 구조 |
| `research/HAIR_WEB_SKIN_EXPERIMENT_EXPLAINED_20260908.md` | 미세조정·해상도·Loss·Backbone 설명 |
| `research/HAIR_WEB_SKIN_IMPROVEMENT_PLAN_20260908.md` | 후속 개선 아이디어 기록 |
| `presentations/DATA_SOURCE_CLASS_TABLE.md` | 원천 클래스와 수량 표 |
| `presentations/INITIAL_DATA_SELECTION_TABLE.md` | 초기 발표 시점의 선정 과정 |
| `presentations/1차.md` | 1차 발표 구성 기록 |

`research/MEDIFLOW_PROJECT_FINAL_REPORT_20260914.md`는 Hair v2 확정 전까지의 장문 종합 기록이다.
현재 수치와 결론은 2026-09-23 최종 종합 문서를 우선한다.

## 4. 노트북 상태

| 노트북 | 상태 | 역할 |
|---|---|---|
| `01_common_dataset_audit_colab.ipynb` | 사용 | 구조·손상·중복·분할 검사 |
| `02_common_original_vs_augmented_colab.ipynb` | 사용 | 원본과 저장 증강본 비교 |
| `03_common_six_experiments_colab.ipynb` | 사용 | B0/B1, 224/256, Loss, 미세조정 비교 |
| `04_hair_three_seed_baseline_colab.ipynb` | 구현 기록 | 현재 결과 생성에는 사용하지 않음 |
| `05_hair_paper_experiment_suite_colab.ipynb` | 구현 기록 | 일괄 실험 참고용, 현재 결과 생성에는 사용하지 않음 |
| `06_hair_supcon_comparison_colab.ipynb` | 실행 완료 | B1·256 기준선과 SupCon 비교 |
| `07_hair_supcon_repeat_seeds_colab.ipynb` | 구현 기록 | 현재 결과 생성에는 사용하지 않음 |
| `08_hair_remaining_methods_screen_colab.ipynb` | 실행 완료 | DINOv2·EfficientNetV2-S·B1 384 선별 |
| `09_hair_b1_384_sam_screen_colab.ipynb` | 실행 완료 | Adam과 SAM 비교 |
| `10_hair_b1_384_final_test_package_colab.ipynb` | 실행 완료 | 최종 Test와 Hair v2 패키징 |
| `11_web_skin_wsdan_attention_colab.ipynb` | 실행 완료 | 기준선과 WS-DAN·PMG·MixStyle 비교, PMG 선정 |
| `12_web_skin_pmg_b1_384_colab.ipynb` | 실행 완료 | PMG·B1·384 Validation 성능과 비용 확인 |
| `13_web_skin_pmg_b0_256_final_test_package_colab.ipynb` | 실행 완료 | 고정된 PMG·B0·256 최종 Test와 후보 v2 패키징 |

`common_colab_notebooks_v1.zip`과 `legacy_notebooks_20260909.zip`은 보관용 묶음이다.

## 5. archive의 의미

`docs/archive/`는 당시 계획, 중간 분석, 검증 과정의 추적 기록이다. 현재 실행 지침으로 쓰지 않는다.

- `planning`: 과거 계획과 보류안
- `analysis`: 당시 결과 분석
- `verification`: 후보와 코드 검증 기록
- `process`: 패키징 과정
- `setup`: 과거 작업 양식

## 6. 결과 폴더 읽는 순서

1. `results/CANDIDATE_INDEX.json`
2. `results/FINAL_MODEL_SUMMARY_20260922.csv`
3. `results/<domain>/candidates/`의 현재 후보 ZIP
4. `results/<domain>/experiments/`의 실험별 원본 기록
5. `results/<domain>/presentation_20260914/`의 발표용 그래프

기존 결과 ZIP, 모델, JSON, CSV와 이미지는 원본 근거이므로 삭제하거나 덮어쓰지 않는다.
