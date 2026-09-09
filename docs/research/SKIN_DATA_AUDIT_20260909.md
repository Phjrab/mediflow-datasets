# Skin 데이터 감사 결과와 정제 계획

기준일: 2026-09-09. 입력은 `skin_processed.zip`, SHA-256은
`3aef92061681b7acbcdca860ec1915a09494034413bc113692d3181af5644d21`이다.

## 확인 결과

10개 클래스와 이미지 수는 균형을 이루고, 손상 이미지와 서로 다른 라벨의 동일 사진은 발견되지 않았다.
Original은 Train 7,300장, Validation 1,000장, Test 700장이다. Augmented는 Train 14,600장이고
Validation과 Test는 Original과 동일한 각각 1,000장과 700장이다. 전체 해상도는 512×512다.

분할 사이에 RGB 픽셀이 완전히 같은 사진 16그룹이 있었다.

| 분할 | 동일 사진 그룹 |
|---|---:|
| Train–Validation | 10 |
| Train–Test | 5 |
| Validation–Test | 1 |

같은 분할 안에서도 Original Train 32그룹(중복 여분 34장), Original Validation 1그룹이 확인됐다.
Augmented Train에는 130그룹(중복 여분 133장)이 있다. Original과 Augmented에 복사된 동일 사진이
함께 조사되므로 두 종류의 분할 간 결과를 합산하지 않는다.

감사 상태는 `issues_found`다. 기존 Original Test Accuracy 0.9942857027053833과 Augmented Test
Accuracy 0.9957143068313599는 과거 보고값으로 보존하지만 독립 Test가 검증된 최종 성능으로 사용하지 않는다.

## 정제 규칙

`skin_clean_v1`은 Original 전체를 다시 모아 RGB 픽셀 해시별로 한 장만 남긴다. 클래스가 다른 동일
픽셀이 발견되면 자동 정제를 중단한다. 클래스별 고유 사진을 클래스명·seed 42·픽셀 해시에서 만든
SHA-256 값으로 고정 정렬한 후 Test 70장, Validation 100장, 나머지를 Train으로 배정한다.

Validation과 Test는 증강하지 않는다. Train 원본마다 기존 Skin 전처리와 동일한 11개 증강 종류 중
하나를 적용해 증강본 한 장을 만든다. 기존 방식의 무작위 원본 재선택은 사용하지 않고 모든 Train
원본을 한 번씩 사용한다. 이 변경은 증강 출처를 완전히 기록하기 위한 필수 변경이다.

결과에는 `split_manifest.csv`, `removed_exact_duplicates.csv`, `augmentation_lineage.csv`, 클래스별
개수, 생성 설정과 소스 해시를 저장한다. 기존 ZIP과 모델은 덮어쓰거나 삭제하지 않는다.

## 정제 실행 결과

실행 ID는 `20260909_063802_6cd88688`이다. 원본 9,000장에서 RGB 픽셀이 같은 중복 파일
51장을 제거해 고유 원본 8,949장을 만들었다. 새 분할은 Train 7,249장, Validation 1,000장,
Test 700장이다. 각 Train 원본에서 증강본을 한 장씩 생성해 Augmented Train은 14,498장이다.
Validation과 Test는 Original과 Augmented에서 같은 평가 사진을 사용한다.

생성 단계 자체 검사 결과는 분할 간 픽셀 중복 0그룹, 라벨 충돌 0그룹, 고유 증강 이미지
7,249장이다. 증강 이미지와 출처 원본의 픽셀 해시가 겹친 사례도 없다. 7,198장은 첫 변환에서
생성됐고, 원본 또는 기존 생성 이미지와 겹친 51장은 한 차례 다시 변환해 고유 이미지를 만들었다.
보고서 ZIP의 CRC와 보고서 내부 파일 해시는 모두 일치했다.

새 데이터 ZIP은 `skin_clean_v1_20260909_063802_6cd88688.zip`이며 크기는 10,528,444,392바이트,
SHA-256은 `db6cf59087b68d4534ff857bdb0eacc1a1a09ed5703772fefc75f038d4570e4b`이다.
이 값은 생성 보고서에 기록된 값이며, 다음 공통 감사에서 Drive의 실제 ZIP을 다시 계산해 대조한다.

## 이후 평가

새 ZIP을 공통 데이터 감사 노트북으로 다시 검사한다. 기계적 검사를 통과하면 공통 Original 대
Augmented 노트북으로 EfficientNet-B0, 224×224, CE, 15 epoch 조건의 두 모델만 새로 학습한다.
Validation으로 모델을 선정하고 선택이 고정된 한 모델만 새 Test에서 평가·패키징한다.

사람·병변·촬영 세션 정보, 유사 장면, 임상 라벨 정확성은 이번 정제로 확인할 수 없다. 실제 USB
현미경 성능도 별도 장비 데이터가 생기기 전까지 미검증이다.
