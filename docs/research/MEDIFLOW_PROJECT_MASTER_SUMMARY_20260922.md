# MediFlow 이미지 분류 프로젝트 최종 종합 정리

기준일: 2026-09-22  
범위: Skin, Web Skin, Hair 공개 데이터 분류 모델  

## 1. 프로젝트가 만든 것

촬영 장비와 영상 범위가 다른 세 문제를 하나의 모델에 섞지 않고 전문 모델로 분리했다.

| 도메인 | 입력 장면 | 역할 | 클래스 수 | 정상 클래스 |
|---|---|---|---:|---|
| Skin | USB 현미경 피부 병변 확대 | 피부 병변 10종 분류 | 10 | 없음 |
| Web Skin | 웹캠 얼굴 정면 | 일상 환경 피부 상태 5종 분류 | 5 | 있음 |
| Hair | USB 현미경 두피 확대 | 두피 증상 5종 분류 | 5 | 없음 |

세 모델은 공개 데이터 후보이며 의료 진단 확정 모델이나 실제 장비 검증 완료 모델이 아니다.

## 2. 전체 진행 흐름

1. Kaggle과 AI Hub에서 피부·두피 데이터를 조사했다.
2. Skin 10-class 분류 모델을 먼저 만들었고 높은 공개 데이터 성능을 확인했다.
3. 얼굴 전체 피부에 병변 검출과 crop을 적용하는 YOLO 구성을 검토했으나, 넓게 퍼진 홍반과
   주변 문맥을 잃을 수 있어 중단했다.
4. 촬영 방식에 따라 Skin, Web Skin, Hair로 역할을 분리했다.
5. 원본과 저장된 증강본을 비교하고, 데이터 중복과 분할 누수를 검사했다.
6. Hair에서 분할 간 완전 동일 이미지가 발견되어 Clean 데이터셋을 재구성했다.
7. 세 도메인에 ImageNet EfficientNet 전이학습과 부분 미세조정을 적용했다.
8. Hair 성능을 더 높이기 위해 SupCon, DINOv2, EfficientNetV2-S, 384 입력, SAM, 앙상블을
   논문 가설에 따라 비교했다.
9. 각 도메인의 최종 공개 데이터 후보를 모델·클래스 순서·전처리·평가 자료와 함께 패키징했다.

## 3. 데이터 출처와 선정

| 도메인 | 원천 | 원천 규모 | MediFlow 초기 선택 | Clean 원본 |
|---|---|---:|---:|---:|
| Skin | AI Hub 피부종양 이미지 합성 데이터 71864 | 15종×1,000=15,000 | 10종×1,000, 초기 분할 9,000 | 8,949 |
| Web Skin | AI Hub 안면부 피부질환 이미지 합성 데이터 71863 | 6종×정면/측면×1,000=12,000 | 지루·측면 제외 정면 5종, 초기 분할 4,500 | 기존 분할 유지 |
| Hair | AI Hub 유형별 두피 이미지 216 | 고유 이미지 101,027 | 5종 중등도·중증 각 2,900=14,500 | 12,536 |

### Skin 클래스

광선각화증, 기저세포암, 보웬병, 사마귀, 지루각화증, 편평세포암, 표피낭종,
피부섬유종, 혈관종, 흑색점이다.

### Web Skin 클래스

출력 순서는 `건선, 아토피, 여드름, 정상, 주사`다. 측면 이미지는 정면과 동일 대상의 짝인지
확실하지 않아 합치지 않았고, 지루는 주사와 홍반·색 변화가 유사하다는 직접 검수를 근거로 제외했다.

### Hair 클래스

출력 순서는 `모낭사이홍반, 미세각질, 비듬, 탈모, 피지과다`다. 모낭홍반/농포는
모낭사이홍반과 시각적으로 유사해 제외했고, 양호·경증은 초기 이상 증상 분류 범위에서 제외했다.

## 4. 데이터 처리와 증강

Original은 원본 Train/Validation/Test로 구성했다. Augmented는 Train에만 증강본을 더하고,
Validation과 Test에는 원본만 사용했다. 따라서 성능 수치는 증강 이미지로 평가한 값이 아니다.

사용한 증강은 밝기·대비·색상 변화, 미세 회전, 좌우 반전, 확대/crop, blur, noise 등이다.
목적은 USB 현미경과 웹캠의 조명, 거리, 초점, 방향 차이를 학습 데이터에 반영하는 것이었다.

Hair 감사에서는 완전 동일 이미지가 Train–Validation 195개, Train–Test 178개,
Validation–Test 16개 발견됐다. 픽셀 해시를 기준으로 중복을 제거하고 분할을 다시 만들었다.
Skin에서도 분할 사이 동일 사진 16그룹과 분할 내부 중복을 확인해 9,000장에서 51장을 제거하고
8,949장의 Clean 원본을 만들었다. Web Skin은 기존 감사에서 완전 동일 중복이 발견되지 않아
기존 분할을 유지했다.

파일 해시로 확인할 수 있는 것은 완전 동일 이미지다. 사람·병변·촬영 세션 식별 정보가 없어
해당 단위의 누수와 매우 비슷한 장면까지 모두 배제했다고 주장하지 않는다.

## 5. 공통 학습 방법

- ImageNet pretrained EfficientNet을 시작점으로 사용했다.
- Stage 1에서는 backbone을 고정하고 새 분류층을 학습했다.
- Stage 2에서는 후반 일부 계층을 낮은 학습률로 미세조정했다.
- 모델 설정은 Validation으로 비교하고, 고정한 후보만 Test에서 최종 평가했다.
- 현재 EfficientNet 저장 모델은 내부에 `Rescaling(1/255)`이 있어 입력은 RGB float32 0–255이며
  외부에서 다시 `/255`하지 않는다.
- 클래스 배열 순서는 모델 출력 계약이므로 결과 JSON의 순서를 그대로 사용한다.

## 6. 도메인별 실험과 결과

### Skin

중복 정제 후 B0·224·CE 조건에서 Original과 Augmented를 비교했다. Augmented의 Validation
Accuracy `0.982`, Macro F1 `0.9820008885656606`가 Original의 `0.973`,
`0.9730455448862949`보다 높아 Augmented 후보를 선택했다. 고정 Test 결과는 Accuracy
`0.9871428571428571`, Macro F1 `0.9871314132317626`이다.

### Web Skin

기존 단순 분류층 학습에서 Augmented Test Accuracy는 약 `0.82`였다. 이후 2단계 미세조정,
224/256 입력, CE/Label Smoothing/Focal, B0/B1, 추가 학습을 비교했다. Validation Accuracy가
가장 높고 사전 동점 규칙을 만족한 B0·256·CE를 선택했다. 고정 Test 400장에서 353장을 맞혀
Accuracy `0.8825`, Macro F1 `0.8813822927981046`을 기록했다.

### Hair

Clean 데이터에서 B0/B1, 224/256, CE/Label Smoothing/Focal과 미세조정 길이를 비교한 뒤,
논문 기반 고급 방법을 추가했다. SupCon은 B1·256 기준선 대비 Validation Macro F1을 약
1.65%p 높였다. 전체 후보 중 B1·384·Label Smoothing 0.05·Adam이 Validation 선두였으며,
최종 Test Accuracy `0.8003194888178914`, Macro F1 `0.8002222282821301`을 기록했다.

## 7. 현재 최종 후보 3개

| 도메인 | 최종 후보 | 입력 | Validation Accuracy | Test Accuracy | Test Macro F1 |
|---|---|---:|---:|---:|---:|
| Hair | EfficientNet-B1·LS 0.05·Adam | 384 | 0.7963258786 | 0.8003194888 | 0.8002222283 |
| Web Skin | EfficientNet-B0·CE | 256 | 0.796 | 0.8825 | 0.8813822928 |
| Skin | EfficientNet-B0·CE·Augmented | 224 | 0.982 | 0.9871428571 | 0.9871314132 |

정확한 후보 ID, 모델 경로와 SHA-256은 `results/CANDIDATE_INDEX.json`이 기준이다.

## 8. 논문 기반 Hair 강화에서 얻은 결론

- SupCon은 같은 클래스 특징을 모으는 가설과 일치하는 개선을 보였고, 목표 혼동도 줄였다.
- DINOv2 고정 encoder는 현재 설정에서 EfficientNet 부분 미세조정보다 낮았다.
- EfficientNetV2-S도 최종 선두가 아니었다.
- 384 입력의 B1이 가장 높아 미세 두피 특징에는 해상도 보존이 유효한 후보임을 확인했다.
- SupCon B1·256과 B1·384의 단순 평균은 B1·384 단일 모델보다 낮았다.
- SAM은 Accuracy가 같고 Macro F1이 소폭 낮아 추가 계산 비용을 정당화하지 못했다.

상세 수치와 논문 링크는 `PAPER_EXPERIMENTS_FINAL_ANALYSIS_20260922.md`에 정리했다.

## 9. 모델 파일과 사용 계약

모델은 `results/<domain>/candidates/` 아래에 ZIP과 압축 해제 폴더로 보존한다. 팀원이 사용할
최소 정보는 다음과 같다.

1. `results/CANDIDATE_INDEX.json`에서 도메인별 current candidate를 확인한다.
2. 함께 들어 있는 `class_names.json`의 순서를 출력 인덱스에 연결한다.
3. `preprocessing.json`의 입력 크기와 dtype을 적용한다.
4. 현재 EfficientNet 모델에는 외부 `/255` 정규화를 적용하지 않는다.
5. 최대 softmax 점수를 실제 정답 확률이나 정상 판정 기준으로 해석하지 않는다.

## 10. 완료 범위와 남은 범위

완료한 것은 공개 데이터 검사, clean 데이터 구성, 전이학습·미세조정, 실험 비교, 최종 후보
선정, Test 평가, 결과 시각화, 모델 패키징과 로컬 입출력 검증이다.

아직 하지 않은 일은 실제 USB 현미경·웹캠 독립 검증, 범위 밖 이미지 거부, Hair/Skin 정상
클래스 추가, 확률 보정, 사람·세션 단위 누수 검증이다. 장비 연결과 LLM/VLM 통합은 현재 작업
범위에서 보류했으며, 팀원이 모델을 연결할 때는 `docs/guides/TEAM_MODEL_QUICKSTART.md`를 따른다.

## 11. 발표·보고서 핵심 이야기

> MediFlow는 촬영 방식이 다른 피부와 두피 영상을 세 전문 분류 문제로 분리했다. 데이터
> 중복을 발견한 뒤 clean 분할을 다시 만들고, ImageNet 전이학습과 부분 미세조정을 공통
> 기준으로 구축했다. 이후 Hair의 유사 클래스 문제를 해결하기 위해 논문의 SupCon, DINOv2,
> EfficientNetV2, SAM과 고해상도 학습을 각각 검증했다. 유명 기법을 모두 채택하지 않고 동일
> Validation에서 실제로 개선된 설정만 남겼으며, 최종 공개 데이터 후보는 Skin 98.71%, Web
> Skin 88.25%, Hair 80.03% Test Accuracy를 기록했다.

## 12. 바로 읽을 파일

- 현재 실행 순서: `docs/ROADMAP.md`
- 논문 실험 분석: `docs/research/PAPER_EXPERIMENTS_FINAL_ANALYSIS_20260922.md`
- 최종 후보 기계 판독 색인: `results/CANDIDATE_INDEX.json`
- 최종 모델 요약표: `results/FINAL_MODEL_SUMMARY_20260922.csv`
- 팀 모델 사용법: `docs/guides/TEAM_MODEL_QUICKSTART.md`
- 문서 상태 색인: `docs/DOCUMENT_STATUS_INDEX_20260922.md`

