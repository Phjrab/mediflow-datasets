# MediFlow 로컬 결과 구조

현재 팀 통합 대상 모델은 각 도메인의 `selected_models` 폴더에 버전별로 정리했다. 정확한 현재
버전, 모델 경로, 설정, 성능과 SHA-256은 [CANDIDATE_INDEX.json](CANDIDATE_INDEX.json)을
기준으로 확인한다. 전체 버전 목록은 [SELECTED_MODELS.md](SELECTED_MODELS.md)를 참고한다.
학습 방법과 v1→v2 개선 이유는 [MODEL_VERSION_COMPARISON.md](MODEL_VERSION_COMPARISON.md)에
표로 정리했다.
표 계산에는 `CURRENT_SELECTED_MODELS.csv`, 버전 비교에는
`SELECTED_MODEL_HISTORY_20260923.csv`를 사용한다. `FINAL_MODEL_SUMMARY_20260922.csv`는 Web Skin
v2 선정 전의 과거 스냅샷이다.
팀원에게 전달할 입력·클래스·실행 예제는 [MODEL_USAGE.md](MODEL_USAGE.md)를 참고한다.

```text
results/
  CANDIDATE_INDEX.json
  SELECTED_MODELS.md
  hair/selected_models/v1, v2/
  web_skin/selected_models/v1, v2/
  skin/selected_models/v1/
  hair/candidates/
  web_skin/candidates/
  skin/candidates/
```

각 `selected_models/vN`에는 실제 모델, `MODEL_INFO.md`, `selection.json`, `class_names.json`,
`preprocessing.json`이 있다. Web Skin v2에는 PMG 출력 처리를 위한 `inference.py`도 있다.

각 `candidates` 폴더는 원본 배포 ZIP과 전체 평가 보고서를 다음 형식으로 보존한다.

```text
public_candidate_vN_<설정>_<실행ID>.zip
public_candidate_vN_<설정>_<실행ID>.zip.sha256
public_candidate_vN_<설정>_<실행ID>/
```

| 도메인 | 현재 후보 | 입력 | 출력 클래스 |
|---|---|---:|---:|
| Hair | EfficientNet-B1 / LS 0.05 / Adam | 384×384 | 5 |
| Web Skin | PMG / EfficientNet-B0 / CE | 256×256 | 5 |
| Skin | EfficientNet-B0 / CE / Augmented | 224×224 | 10 |

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

세 후보 모두 공개 데이터 기준이며 실제 장비 데이터 검증 전 상태다. EfficientNet 모델 내부에
`Rescaling(1/255)`이 있으므로 입력은 RGB float32 0–255를 사용하고 외부 `/255`를 적용하지 않는다.
Web Skin 후보는 네 PMG logit 출력을 합산한 뒤 softmax를 적용해야 하므로 후보 ZIP의
`inference.py` 또는 공통 재현 모듈을 사용한다.
