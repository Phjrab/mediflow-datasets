# MediFlow 로컬 결과 구조

현재 배포 검토 대상은 각 도메인의 `candidates` 폴더에 있다. 정확한 모델 경로, 설정, 성능과
SHA-256은 [CANDIDATE_INDEX.json](CANDIDATE_INDEX.json)을 기준으로 확인한다.
팀원에게 전달할 입력·클래스·실행 예제는 [MODEL_USAGE.md](MODEL_USAGE.md)를 참고한다.

```text
results/
  CANDIDATE_INDEX.json
  hair/candidates/
  web_skin/candidates/
  skin/candidates/
```

각 `candidates` 폴더는 같은 형식을 사용한다.

```text
public_candidate_vN_<설정>_<실행ID>.zip
public_candidate_vN_<설정>_<실행ID>.zip.sha256
public_candidate_vN_<설정>_<실행ID>/
```

| 도메인 | 현재 후보 | 입력 | 출력 클래스 |
|---|---|---:|---:|
| Hair | EfficientNet-B1 / LS 0.05 / Adam | 384×384 | 5 |
| Web Skin | EfficientNet-B0 / CE | 256×256 | 5 |
| Skin | EfficientNet-B0 / CE / Augmented | 224×224 | 10 |

`1_training`, `experiments`, `archives`, `selected_models`는 과거 결과와 실험 근거를
보존하는 폴더다. 실제 통합에서 사용할 후보를 찾을 때는 `candidates`와
`CANDIDATE_INDEX.json`을 사용한다. 기존 결과 파일은 후보 폴더를 만들기 위해 이동하거나
삭제하지 않았다.

세 후보 모두 공개 데이터 기준이며 실제 장비 데이터 검증 전 상태다. EfficientNet 모델 내부에
`Rescaling(1/255)`이 있으므로 입력은 RGB float32 0–255를 사용하고 외부 `/255`를 적용하지 않는다.
