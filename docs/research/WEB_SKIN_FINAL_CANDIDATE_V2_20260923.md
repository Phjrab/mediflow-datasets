# Web Skin 최종 공개 데이터 후보 v2

## 결론

Validation과 실행 비용을 함께 검토해 고정한 `PMG·EfficientNet-B0·256·CE·Adam` 모델을
Test 400장에서 한 번 평가하고 배포 후보 v2로 패키징했다. Test Accuracy는 `0.915`, Macro
F1은 `0.9141049081029712`다. 기존 공개 후보 v1은 과거 기록으로 보존하며 현재 통합 기준은
`results/CANDIDATE_INDEX.json`의 v2 후보다.

## 고정 조건과 결과

| 항목 | 값 |
|---|---|
| 후보 ID | `public_candidate_v2_pmg_b0_256_ce_20260922_235840_093d10de` |
| 방법 | PMG adaptation + ImageNet EfficientNet-B0 |
| 입력 | 256×256 RGB float32, 0–255 |
| 학습 | augmented train, Stage 1 15 + Stage 2 10 epoch |
| 선택 기준 | Validation과 배포 비용의 균형 |
| Validation | Accuracy `0.85`, Macro F1 `0.8473279632397033`, 500장 |
| Test | Accuracy `0.915`, Macro F1 `0.9141049081029712`, 400장 |
| 파라미터 | `8,872,375` |
| 정상 클래스 | 있음 |

## Test 클래스별 결과

| 클래스 | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| 건선 | 0.9041095890 | 0.825 | 0.8627450980 | 80 |
| 아토피 | 0.8974358974 | 0.875 | 0.8860759494 | 80 |
| 여드름 | 0.8780487805 | 0.900 | 0.8888888889 | 80 |
| 정상 | 0.9638554217 | 1.000 | 0.9815950920 | 80 |
| 주사 | 0.9285714286 | 0.975 | 0.9512195122 | 80 |

Test에서 400장 중 366장을 맞혔다. 건선 Recall이 `0.825`로 가장 낮았으며 건선 80장 중
6장은 아토피, 6장은 여드름, 2장은 주사로 분류됐다. 정상 80장은 모두 정상으로 분류됐지만,
다른 클래스 3장이 정상으로 분류됐으므로 정상 판정이 완전하다는 뜻은 아니다.

## 패키지와 무결성

- 로컬 ZIP: `results/web_skin/candidates/public_candidate_v2_pmg_b0_256_ce_20260922_235840_093d10de.zip`
- 모델: `web_skin_pmg_model.keras`
- ZIP SHA-256: `38bd47493ec06bcf2be351a7d12b8f889489fd5d8205e4fcb39557a400471db9`
- 모델 SHA-256: `83e659dd09a9355135ee0de0197037e81ece962777f59dedae8d9a37b49a3fc4`
- ZIP CRC: 통과
- manifest 파일 해시: 18개 모두 일치
- 로컬 모델 load와 dummy forward: 통과
- 모델 계약: 입력 `(None, 256, 256, 3)`, 내부 출력 4개×5 logits
- 공통 추론 모듈 검사: `candidate_reproduction_v2` 샘플 2장 통과

PMG 모델 파일 자체는 네 개의 logit 배열을 출력한다. 서비스에서는 패키지의 `inference.py`
또는 공통 `candidate_reproduction.py`처럼 네 출력을 더한 후 softmax를 적용해야 한다. 모델에
softmax를 개별 적용하거나 첫 번째 출력만 사용하면 학습 당시 평가와 다른 결과가 된다.

## 전처리와 클래스 계약

클래스 순서는 `건선, 아토피, 여드름, 정상, 주사`다. 이미지를 RGB 3채널로 읽고 TensorFlow
bilinear로 256×256에 맞춘다. 입력은 float32 0–255이며 모델 내부에 `Rescaling(1/255)`이
있으므로 외부 `/255`를 적용하지 않는다. EXIF 자동 회전, crop, padding과 별도 ImageNet
정규화는 현재 재현 계약에 포함되지 않는다.

## 해석 한계

이 결과는 공개 데이터 기준이다. 실제 웹캠 환자 사진은 평가하지 않았고 범위 밖 이미지 거부,
확률 보정, 사람·병변·촬영 세션 단위 누수 검증도 완료하지 않았다. softmax 점수를 정답 확률이나
의료적 진단 확률로 해석하지 않는다.
