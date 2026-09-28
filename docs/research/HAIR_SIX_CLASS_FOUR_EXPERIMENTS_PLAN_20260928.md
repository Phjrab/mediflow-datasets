# Hair 6클래스 원본·증강 × 두 모델 실험 계약

상태: **Colab 코드 준비, 실제 데이터셋 생성·학습 미실행**.
실행 노트북: `notebooks/17_hair_six_class_four_experiments_colab.ipynb`.
기존 5클래스 모델과 데이터를 덮어쓰지 않는다.

## 데이터 구성

- 입력은 감사가 끝난 Hair Clean v1 원본 12,536장과 사용자가 여섯 증상값 0을
  확인한 새 ‘양호’ JPG 534장이다. ZIP SHA-256을 검사해 감사 때 사용한 파일과
  동일할 때만 진행한다. 새 ZIP에는 파일별 라벨 JSON이 없다.
- 기존 5클래스의 Train/Validation/Test 분할은 유지한다. 원본 사진 이름은
  `탈모_01164.jpg`처럼 새 번호로 저장돼 사람·세션 ID를 복구할 수 없다.
- ‘양호’는 파일명 첫 숫자 ID별로 묶어 약 80:10:10으로 새 분할한다. 실제
  사진·ID 수와 각 사진의 원천 경로를 `dataset_contract.json`과
  `image_manifest.csv`에 기록한다. 새 이미지가 기존 원본과 완전히 같은 경우
  데이터셋 생성을 중단한다.
- 원본 실험의 Train은 원본만 쓴다. 증강 실험에서는 기존 5개 클래스의 Train
  원본 중 클래스당 약 50%를 골라 각각 1장씩 약하게 변형한다. 원본이 적은
  ‘양호’ Train은 원본당 5장씩 변형한다. 밝기·대비·색감·±10° 회전·좌우 반전·
  약한 블러만 적용하며, 각 증강본의 원천과 설정을 기록한다. Validation/Test
  이미지에는 증강을 적용하지 않는다. 증강본은 새 사람이 아니다.

## 네 실험

| ID | 모델 | 크기 | Loss | Train |
|---|---|---:|---|---|
| `b0_256_original` | EfficientNet-B0 | 256 | CE | 원본 |
| `b0_256_augmented` | EfficientNet-B0 | 256 | CE | 원본+증강 |
| `b1_384_original` | EfficientNet-B1 | 384 | Label Smoothing 0.05 | 원본 |
| `b1_384_augmented` | EfficientNet-B1 | 384 | Label Smoothing 0.05 | 원본+증강 |

공통 고정 조건은 새 6클래스 데이터 버전, 각 원본의 분할, 클래스 순서,
ImageNet 사전학습, seed 42, batch 32, head-only 15 epoch와 마지막 30개 층
부분 미세조정 15 epoch다. 각 실험은 별도 새 모델로 시작한다. 같은 모델의
원본·증강 비교에서는 Train 이미지와 해당 이미지 수로 계산한 완만한 클래스
가중치가 달라진다. B0와 B1 비교에는 구조·해상도·Loss가 함께 바뀌므로
단일 요소의 효과로 설명하지 않는다.

각 실험의 가장 좋은 체크포인트는 Validation Accuracy로 저장한다. 네 후보는
Validation Macro F1, 동률이면 Accuracy로 선정한다. 네 후보의 Validation
Accuracy·Macro F1·‘양호’ Recall·클래스별 F1·학습곡선과 학습 시간을 보존한다.
선정 모델 하나만 고정 Test에서 평가하고 혼동행렬을 만든다. 5클래스 v1/v2
수치를 새 6클래스 수치와 직접 성능 향상률로 비교하지 않는다.

## 남는 한계

기존 5클래스의 사람·촬영 세션 간 중복은 파일명 변경으로 확인하지 못했다.
두 ZIP 사이 바이트와 RGB 픽셀 완전 일치는 0장이지만 유사 사진 및 사람
겹침이 없다고 단정할 수 없다. 작은 ‘양호’ Validation/Test와 증강 원본의
높은 반복 비율도 최종 결과 해석에 함께 기록한다.
