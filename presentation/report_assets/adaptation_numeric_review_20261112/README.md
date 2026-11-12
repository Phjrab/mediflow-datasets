# 어댑테이션 조절값·장비 유사성·성능 수치 자료

[전체 설명 보고서](../../../docs/research/camera_adaptation/ADAPTATION_NUMERIC_COMPARISON_20261112.md) ·
[엑셀 비교표](MediFlow_Adaptation_Numeric_Review_20261112.xlsx).

| 자료 | 내용 |
| --- | --- |
| [verification.json](verification.json) | 출처 37개 해시·엑셀 수식·원문 성능값·전체 테스트·보존 확인 |
| [numeric_review.json](numeric_review.json) | 17프로필·21학습·221설정·107W1·21특징 기록(미측정3 포함)·81Test·원본 해시 |
| [device_gaps.csv](device_gaps.csv) | 분포 거리 전후·차이·원본 대비 비율·감소율·측정 체계 |
| [distributions.csv](distributions.csv) | 원본·변환·실제 장비 평균·분산·분위수·표본 수 |
| [features.csv](features.csv) | MMD²/CORAL 원문·기준 모델·표본·미측정 표시 |
| [source_test.csv](source_test.csv) | Source Test 81개 기록·정답 수·Accuracy·Macro-F1·기준 대비 차이 |
| [webcam_phone.csv](webcam_phone.csv) | 같은 볼 ROI에서 웹캠·휴대폰 값 비교 |
| [거리 비율 그림](device_gap_ratios.png) | 1보다 작으면 가까워지고 1보다 크면 멀어지는 항목별 비교 |
| [실제 밝기 분포](brightness_distributions.png) | 원본·4차·실제 사진의 중앙값/IQR/5–95분위 |
| [특징 거리 그림](feature_gap_ratios.png) | 같은 진단의 적응 전 모델 기준 MMD/CORAL 비율 |

W1 감소율은 이미지 유사도나 정확도 %가 아니다. 사람·질환·부위·기기 차이가 섞여 있고 Target 질환 정답은 없다.
강제변환 표본 재생은 학습의 변환 확률 0.5 혼합분포와 다르다. 코드·모델·Test·강도는 바꾸지 않았다.
Excel에서 숫자 형식의 거리는 표시용 6자리·변화율 2자리다. 원문 JSON/CSV는 전체 자릿수를 보존한다.
Accuracy/F1은 원문 자릿수의 Acc/F1 텍스트도 병기해 Excel의 15자리 제한과 미리보기의 자동 반올림을 피했다.

저장된 원문 수치를 다시 추출하는 코드는 [adaptation_numeric_review.py](../../../src/mediflow_datasets/adaptation_numeric_review.py),
검사는 [test_adaptation_numeric_review.py](../../../tests/test_adaptation_numeric_review.py)에 있다.
저장소에서 `python -m mediflow_datasets.adaptation_numeric_review --output 새_출력_폴더`로 실행한다.
기본값은 기존 폴더 덮어쓰기를 거부한다. 이 코드는 기존 기록을 모으며 새 이미지 분석·학습·Test를 실행하지 않는다.

2026-11-12 폴더 정리에서 엑셀을 이 폴더로 이동했다. 엑셀 바이트와 원문 수치는 같다. `verification.json`은 최초 작성 당시 기록이며, 링크 변경 후 보존 검사는 [최신 정리 검증](../../../docs/research/maintenance/PROJECT_STORAGE_CLEANUP_20261112.json)에 있다.
