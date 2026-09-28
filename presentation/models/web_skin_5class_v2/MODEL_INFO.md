# Web Skin selected model v2

- 상태: **현재 사용 모델**
- 원본 후보 ID: `public_candidate_v2_pmg_b0_256_ce_20260922_235840_093d10de`
- 구조: PMG with EfficientNet-B0
- 입력: 256×256 RGB float32, 0–255
- Loss: Categorical Crossentropy
- Validation Accuracy: `0.85`
- Test Accuracy: `0.915`
- Test Macro F1: `0.9141049081029712`
- 출력 처리: `sum_logits_softmax`
- 모델 SHA-256: `83e659dd09a9355135ee0de0197037e81ece962777f59dedae8d9a37b49a3fc4`

## 선정 이유

논문 기반 Web Skin 실험에서 PMG·B0·256이 기존 v1보다 Validation 성능을 높였고, B1·384보다 가벼우면서 실제 배포 비용과 성능의 균형이 좋아 현재 후보로 선정했다.

## 버전 설명

현재 버전이다. 모델의 네 logit 출력을 합산한 뒤 softmax를 적용해야 한다.

## 사용 파일

- 모델: `web_skin_model.keras`
- 클래스 순서: `class_names.json`
- 전처리 계약: `preprocessing.json`
- 구조화된 기록: `selection.json`
- PMG 출력 변환: `inference.py`

전체 원본 보고서와 배포 ZIP은 `results/web_skin/candidates/public_candidate_v2_pmg_b0_256_ce_20260922_235840_093d10de`에서 보존한다.
