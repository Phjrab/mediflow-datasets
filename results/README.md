# MediFlow 로컬 결과 구조

연구 전체 과정·기본~4차 결과·후속 후보·자료 위치는 [프로젝트 정리](../docs/PROJECT_SUMMARY_20261111.md)를 먼저 확인한다.

이 폴더는 **선정 모델과 평가 원본을 함께 보존**한다. 모델을 찾거나 팀원에게 전달할 때는 아래 순서로 읽는다.

## 11월 실제 촬영 환경 적응 결과

[피부 4차·얼굴 4차·두피 2차 사용 ZIP](camera_adaptation_packages_20261111/README.md)을 패키징하고
실제 로드·원본 예측 일치·압축 해제본의 단독 실행을 확인했다. 각 ZIP에 클래스·전처리·사용 예시·
원본 학습 및 평가 기록이 있다. 기존 공식 후보 색인과 공통 추론 경로는 유지했다.

[4차 추가 Test 결과](camera_adaptation_stage4_test_20261111/README.md)를 회수했다. 신규 15개·이전 36개 재사용의 51개 비교를 검증했다. 모델 선정에 Test를 사용하지 않았고 실제 장비 질환 Accuracy는 미측정이다.

[조정 전 4차 결과](camera_adaptation_stage4_before_refit_20261111/README.md)를 추가했다. 피부·얼굴은 이번 비교에서 4차, 두피는 2차가 선택됐다. 재설정 4차는 실행하지 않고 보관한다. [기본~4차 공통 Test](../docs/research/camera_adaptation/ADAPTATION_STAGE4_FINAL_TEST_20261111.md)는 완료한 실행의 재현용이다.

| 완료한 작업 | 저장 위치 | 설명 |
| --- | --- | --- |
| 1차 6회 | [camera_adaptation_stage1_20261110](camera_adaptation_stage1_20261110) | 초기·선정 모델의 조명 적응, 분야별 후보 선정 |
| 2차 3회 | [camera_adaptation_stage2_20261110](camera_adaptation_stage2_20261110) | 선택한 출발 모델에서 색·미세 대비·흐림 확장 |
| 3차 3회 | [camera_adaptation_stage3_20261110](camera_adaptation_stage3_20261110) | 같은 출발점·확장 변환에 MMD·CORAL 결합 손실 |
| 최종 Source Test 36비교 | [camera_adaptation_final_test_20261110](camera_adaptation_final_test_20261110) | 학습 전·1차·2차·3차 × 원본·조명·확장 조건 |

[과정·결과 해석](../docs/research/camera_adaptation/CAMERA_ADAPTATION_RETROSPECTIVE_20261110.md), [최종 Test 분석](../docs/research/camera_adaptation/ADAPTATION_FINAL_TEST_RESULT_REVIEW_20261110.md), [완료 실험 노트북](../notebooks/camera_adaptation/completed/README.md)을 함께 확인한다. 조정 전 4차는 완료했고 재설정 4차는 사용자 결정으로 미실행 보관했다. 실제 촬영 자료에는 확인된 질환 정답이 없어 Target Accuracy를 측정하지 않았다.

## 기존 공개 데이터 모델과 보고서

| 찾을 것 | 기준 파일 |
|---|---|
| 선정 모델 7개의 실제 경로·현재 사용 상태 | [선정 모델 경로](SELECTED_MODEL_PATHS.md) |
| 기존 세 분류기의 프로그램 연결 경로 | [CANDIDATE_INDEX.json](CANDIDATE_INDEX.json) — Hair는 **5클래스** |
| Hair 6클래스 연구 후보를 포함한 다섯 현재 선정본의 경로·해시·지표 | [PUBLIC_MODEL_SELECTION_20260928.json](PUBLIC_MODEL_SELECTION_20260928.json) |
| 선정 이유와 성능·실행 비용 비교 | [최종 선정](FINAL_MODEL_SELECTION_20260928.md), [v1→v2 비교](MODEL_VERSION_COMPARISON.md) |
| 입력·출력·오류 처리 및 전달 방법 | [모델 사용법](MODEL_USAGE.md), [재현 기준](REPRODUCTION_GUIDE.md) |

`CURRENT_SELECTED_MODELS.csv`와 `SELECTED_MODEL_HISTORY_20260923.csv`는 비교표 계산 자료다.
`FINAL_MODEL_SUMMARY_20260922.csv`는 Web Skin v2 선정 **전**의 과거 스냅샷으로 읽는다.
[저장 평가 기록 대조](EVALUATION_RECORD_AUDIT_20260927.json)는 기존 예측·모델 해시의 확인이며 새 이미지 추론이 아니다.
프로젝트 단계와 전체 연구 설명은 [문서 안내](../docs/README.md)를 따른다.

2026-10-26 LLM·VLM 착수 전 재점검의 실제 파일·저장 지표·샘플 추론·테스트 근거는
[사전 점검 결과](pre_llm_vlm_audit_20261026/README.md)에 추가했다. 기존 평가 원본은 유지한다.
팀원 LLM·VLM 저장소의 브랜치·코드·공개 집계와 로컬 검사 근거는
[저장소 검토](team_llm_vlm_review_20261026/README.md)에 있다.

2026-09-28 저장 모델 34개 실행 시간 측정의 원본 ZIP, 수치 CSV/JSON, 발표용 그림 12장은
[runtime_benchmark_20260928_051423](runtime_benchmark_20260928_051423/README.md)에 정리했다.
수치 해석과 비교 가능 범위는 [측정 결과 검토](../docs/research/public_data/RUNTIME_BENCHMARK_REVIEW_20260928.md)를 따른다.

```text
results/
  CANDIDATE_INDEX.json
  SELECTED_MODEL_PATHS.md
  hair/selected_models/v1, v2/
  hair/selected_models/6class_performance_v1, 6class_light_v1/
  web_skin/selected_models/v1, v2/
  skin/selected_models/v1/
  hair/candidates/
  web_skin/candidates/
  skin/candidates/
```

각 `selected_models/vN`에는 실제 모델, `MODEL_INFO.md`, `selection.json`, `class_names.json`,
`preprocessing.json`이 있다. Web Skin v2에는 PMG 출력 처리를 위한 `inference.py`도 있다.
Hair 6-class의 두 폴더에도 모델과 사용 계약을 같은 형식으로 복사했다. 원본 ZIP은
`hair/candidates/`에 그대로 보존하고 각각 SHA-256 확인 파일을 추가했다.

2026-09-28 로컬 점검에서 `.keras` 파일은 18개, 파일 내용 기준 서로 다른 모델은 13개였다.
차이 5개는 Hair 5클래스 v1/v2, Skin v1, Web Skin v1/v2의 `candidates/` 원본과
`selected_models/` 사용 사본이 같은 해시로 존재하기 때문이다. **중복을 이유로 어느 쪽도
삭제하지 않는다.** 18개 모두 Keras 아카이브 구조 검사를 통과했고, 내용이 다른 13개 모델은
TensorFlow 2.20.0에서 `compile=False`로 실제 로드되어 각 출력 차원을 확인했다. 결과 ZIP 18개는
CRC 검사에 통과했고, 후보 ZIP의 SHA-256 확인 파일 7개는 모두 해당 ZIP과 일치했다.
이 검사는 새 이미지 추론·정확도 재평가를 뜻하지 않는다.

각 `candidates` 폴더는 원본 배포 ZIP과 전체 평가 보고서를 다음 형식으로 보존한다.

```text
public_candidate_vN_<설정>_<실행ID>.zip
public_candidate_vN_<설정>_<실행ID>.zip.sha256
public_candidate_vN_<설정>_<실행ID>/
```

| 도메인 | 현재 후보 | 입력 | 출력 클래스 |
|---|---|---:|---:|
| Hair 5-class (기존 통합) | EfficientNet-B1 / LS 0.05 / Adam | 384×384 | 5 |
| Web Skin | PMG / EfficientNet-B0 / CE | 256×256 | 5 |
| Skin | EfficientNet-B0 / CE / Augmented | 224×224 | 10 |
| Hair 6-class (별도 성능 우선 후보) | EfficientNet-B1 / LS 0.05 / Adam | 384×384 | 6 |
| Hair 6-class (별도 경량 대안) | EfficientNet-B0 / CE / Augmented | 256×256 | 6 |

`1_training`, `experiments`, `archives`는 학습·비교·보관 자료다. 실제 통합에서는
`selected_models`와 `CANDIDATE_INDEX.json`을 사용하고, 재현용 원본 ZIP과 상세 보고서는
`candidates`에서 확인한다. 기존 결과 파일은 이동하거나 삭제하지 않았다.

2026-09-24 Web Skin MedSigLIP-448 Frozen Linear 실험은 Validation Accuracy `0.824`,
Macro F1 `0.8195774288599683`으로 현재 PMG v2보다 낮아 미채택했다. 재현 파일은
`web_skin/experiments/web_skin_medsiglip_linear_20260924_072442_f7176e07`, 원본 보고서 ZIP은
`web_skin/archives`에 보존한다. 이 실험은 `selected_models`에 포함하지 않으며 Test도 실행하지
않았다.

2026-09-24 Hair MedSigLIP-448 Frozen Linear 실험은 Validation Accuracy
`0.7699680511182109`, Macro F1 `0.7687534259685547`로 현재 Hair v2보다 낮아 미채택했다.
재현 파일은 `hair/experiments/hair_medsiglip_linear_20260924_081708_9479fbb4`, 원본 보고서
ZIP은 `hair/archives`에 보존한다. 이 실험은 `selected_models`에 포함하지 않으며 Test도 실행하지
않았다.

위 다섯 선정본 모두 공개 데이터 기준이며 실제 장비 데이터 검증 전 상태다. EfficientNet 모델 내부에
`Rescaling(1/255)`이 있으므로 입력은 RGB float32 0–255를 사용하고 외부 `/255`를 적용하지 않는다.
Web Skin 후보는 네 PMG logit 출력을 합산한 뒤 softmax를 적용해야 하므로 후보 ZIP의
`inference.py` 또는 공통 재현 모듈을 사용한다.
