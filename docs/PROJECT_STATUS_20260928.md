# MediFlow 현재 상태와 자료 찾기

기준일: 2026-09-28. 이 문서는 팀원 GitHub 검토 이후 로컬 연구 결과까지 이어지는 **현재 상태의 입구**다. 당시 판단과 수치는 작성 시점별 원본 문서·결과 파일에 보존하며, 이 페이지에서 과거 실험을 새로 수행했다고 뜻하지 않는다.

## 지금 무엇이 완료됐나

1. [팀원 키오스크 저장소 분석](research/KIOSK_CORE_REVIEW_20260913.md): 당시 고정 커밋에서 웹·안구·LLM 코드와 세 피부·두피 분류기의 연결 지점을 검토했다. **분석·연결 계획**이며 키오스크 통합 완료 기록은 아니다.
2. [데이터·초기 학습부터 논문 실험까지의 이유와 결과](research/ALL_EXPERIMENTS_RATIONALE_AND_RESULTS_20260924.md): 정제, 증강, 미세조정, 해상도, Loss, Backbone, SupCon·PMG·SAM·MedSigLIP 등의 가설과 채택 여부를 기록했다.
3. [Hair 6클래스 실험](research/HAIR_5CLASS_6CLASS_EXPERIMENT_ANALYSIS_20260928.md): 기존 5클래스를 보존하고 `양호`를 추가한 별도 과제의 네 학습 조건, 두 패키지와 남은 데이터 검증을 기록했다.
4. [저장 모델 34개 실행 시간 측정](research/RUNTIME_BENCHMARK_REVIEW_20260928.md): Skin 4, Web Skin 12, Hair 5클래스 14, Hair 6클래스 4개를 Colab에서 측정했다. 실제 장비의 지연 시간 측정은 아직 아니다.
5. [공개 데이터 후보 종합 선정](../results/FINAL_MODEL_SELECTION_20260928.md): 분류 성능과 같은 환경의 실행 비용을 함께 해석했다. 선정 모델 파일·클래스 순서·원본 ZIP은 [모델 색인](../results/PUBLIC_MODEL_SELECTION_20260928.json)에서 찾는다.

## 현재 보존한 선정 모델

| 과제 | 역할 | 모델 폴더 | 입력 |
|---|---|---|---:|
| Skin 10클래스 | 기존 통합 후보 | [`skin/v1`](../results/skin/selected_models/v1/MODEL_INFO.md) | 224 |
| Web Skin 5클래스 | 기존 통합 후보 | [`web_skin/v2`](../results/web_skin/selected_models/v2/MODEL_INFO.md) | 256 |
| Hair 5클래스 | 기존 통합 후보 | [`hair/v2`](../results/hair/selected_models/v2/MODEL_INFO.md) | 384 |
| Hair 6클래스 | 성능 우선 연구 후보 | [`6class_performance_v1`](../results/hair/selected_models/6class_performance_v1/MODEL_INFO.md) | 384 |
| Hair 6클래스 | 경량 대안 | [`6class_light_v1`](../results/hair/selected_models/6class_light_v1/MODEL_INFO.md) | 256 |

기존 프로그램용 [CANDIDATE_INDEX.json](../results/CANDIDATE_INDEX.json)은 Skin, Web Skin, **Hair 5클래스** 세 경로를 유지한다. Hair 6클래스는 [별도 색인](../results/PUBLIC_MODEL_SELECTION_20260928.json)의 `hair_6class_primary` 또는 `hair_6class_light`를 명시적으로 선택해야 한다. 5클래스와 6클래스는 출력 계약과 평가 과제가 달라 성능 수치를 개선 전후처럼 직접 비교하지 않는다. 모델 입력·출력과 오류 처리는 [MODEL_USAGE.md](../results/MODEL_USAGE.md)를 따른다.

## 파일을 찾는 순서

- **현재 해야 할 일:** [ROADMAP.md](ROADMAP.md). 모델 구조 비교와 실제 하드웨어 측정은 계획이며 완료 표시가 아니다.
- **발표·보고서용 전체 흐름:** [2026-09-28 최종 상세 보고서](FINAL_DETAILED_REPORT_20260928.md)에 초기 수집부터 Hair 6클래스·실행 비용·향후 계획까지 한 파일로 정리했다.
- **모델만 전달할 때:** [선정 모델 7개 경로](../results/SELECTED_MODEL_PATHS.md)와 [모델 사용법](../results/MODEL_USAGE.md)을 따른다. 해당 모델의 `class_names.json`·`preprocessing.json`·SHA-256도 함께 전달한다.
- **Colab 실험을 다시 볼 때:** [노트북 안내](../notebooks/사용안내.md). 로컬 노트북 20개 중 공통 3개와 이후 실험·평가 노트북의 역할을 구분한다.
- **원본 근거:** `results/<domain>/candidates/`는 배포 ZIP·평가 보고서, `experiments/`는 선별 실험, `selected_models/`는 실제 선정본이다. 오래된 기록을 최신 지표로 덮어쓰지 않는다.

## 아직 확인되지 않은 것

- 실제 USB 현미경·웹캠 사진에서의 분류 성능, 최종 장비의 지연 시간·메모리·전력은 측정되지 않았다.
- Hair 6클래스의 사람·촬영 세션 단위 독립성과 신규 라벨 근거는 후속 검증 과제다.
- 팀원 키오스크와 LLM/VLM에 세 분류기를 실제 연결하거나 배포한 기록은 없다. 2026-09-13 GitHub 분석은 당시 커밋을 기준으로 한 검토이며 현재 원격 저장소의 상태를 자동으로 보증하지 않는다.

각 후보의 Test 값과 Colab 측정값은 [종합 선정 문서](../results/FINAL_MODEL_SELECTION_20260928.md)에 원본 출처와 함께 있다. 이 페이지는 경로 안내이며 새로운 평가 결과가 아니다.
