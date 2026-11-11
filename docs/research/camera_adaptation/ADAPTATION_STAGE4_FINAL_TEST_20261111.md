# 4차만 추가 Source Test · 기존 기본~3차 결과 재사용 · 2026-11-11

## 사용자 결정과 현재 단계

사용자는 조정 전 4차만 채택하고 재설정 4차는 실행하지 않기로 했다. 완료한 4차의 설정 식별값은 `492ee01afb4579b6402fca29459c59d56618243589642fc022db18bbbe45a165`다. 미실행 재설정본 `e6bc85bc0a7d`는 보관 자료로 분리했다. 이번 작업은 새 학습이 아니라 고정한 모델들의 기존 Test 비교다.

**실제 Colab 실행·결과 회수·검증을 완료했다.** [4차 추가 Test 결과 분석](ADAPTATION_STAGE4_TEST_RESULT_REVIEW_20261111.md)과 [원본 결과](../../../results/camera_adaptation_stage4_test_20261111/README.md)를 확인한다. 아래 준비·실행·오류 수정은 완료 실험의 재현 기록이다. 이전 기본~3차 Test는 이미 확인한 자료이므로 이번 비교를 새로운 독립 미관측 Test 검증이라고 표현하지 않는다. Test로 변환 강도나 모델을 다시 선택하지 않는다.

## 정리한 파일 위치

| 역할 | 파일·폴더 |
| --- | --- |
| 완료한 Test 재현 노트북 | [MediFlow_Adaptation_Stage4_Test_20261111.ipynb](../../../notebooks/camera_adaptation/completed/MediFlow_Adaptation_Stage4_Test_20261111.ipynb) |
| Test 평가 계획 | [adaptation_stage4_test_20261111.json](../../../configs/adaptation_stage4_test_20261111.json) |
| 완료한 4차 재현 노트북 | [completed/MediFlow_Adaptation_Stage4_20261111.ipynb](../../../notebooks/camera_adaptation/completed/MediFlow_Adaptation_Stage4_20261111.ipynb) |
| 완료한 4차 설정 | [camera_revised_20261111.json](../../../configs/camera_revised_20261111.json) |
| 완료한 4차 모델·학습 결과 | [camera_adaptation_stage4_before_refit_20261111](../../../results/camera_adaptation_stage4_before_refit_20261111/README.md) |
| 미실행 재설정 노트북 | [archive/MediFlow_Camera_Refit_Not_Run_20261111.ipynb](../../../notebooks/camera_adaptation/archive/MediFlow_Camera_Refit_Not_Run_20261111.ipynb) |
| 미실행 재설정 설정 | [configs/archive/camera_refit_not_run_20261111.json](../../../configs/archive/camera_refit_not_run_20261111.json) |
| 미실행 재설정 분석 근거 | [CAMERA_REFIT_20261111.md](CAMERA_REFIT_20261111.md) |

재설정 노트북과 설정은 바이트를 바꾸지 않고 옮겼다. 완료한 4차는 학습 직전 보존본을 복사해 `completed`에 연결했고 기존 보존본도 유지했다. 모델·원본 결과·이미지·기존 Test는 이동하거나 덮어쓰지 않았다. 결과 폴더의 `before_refit`은 완료한 4차의 출처를 구분하는 이름으로 유지한다.

## 고정한 비교

피부 700장, 얼굴 400장, 두피 1,252장의 원본 Test 목록·정답·클래스 순서·데이터 ZIP 해시를 이전 평가와 동일하게 고정했다. 출발 모델은 피부 초기 B0·224, 얼굴 선정 PMG·B0·256, 두피 선정 B1·384 5클래스다.

새로 추론하는 모델은 분야별 4차 모델 하나씩, 총 3개다. 아래 다섯 조건으로 3분야 × 1모델 × 5조건 = **15개만 새로 평가**한다. 기본·1·2·3차의 기존 36개 Test 결과는 검증된 원본 JSON을 노트북에 포함해 재사용한다. 기존 모델은 다시 로드하거나 추론하지 않는다. 앞서 준비한 미실행 75개 비교안은 이 구성으로 대체했다.

| 조건 | 내용 |
| --- | --- |
| clean | 원본 Test |
| lighting | 기존 1차 조명 변환 |
| expanded | 기존 2차 확장 변환 |
| stage4_lighting | 완료한 4차 조명 변환 |
| stage4_expanded | 완료한 4차 확장 변환 |

원본·기존 조명·기존 확장은 이전 Test와 같은 사진·전처리·변환 코드·프로필을 사용한다. 변환 난수는 기존과 같은 seed 1707이며 사진 경로로 고정한다. 모든 변환 평가는 해당 조건을 전체 Test에 적용하고 학습의 50% 적용 확률을 사용하지 않는다. 이는 변환 강도를 늘리는 설정이 아니다. 두피의 4차 lighting과 expanded는 실제로 같으므로 독립된 두 종류의 근거로 세지 않는다.

새 4차 조명·확장 두 조건은 기본~3차에서 평가한 기록이 없다. 해당 조건에서는 4차 점수만 기록하고 기본 대비 변화량을 비워 둔다. 이전 모델의 원본 또는 다른 변환 점수를 가져와 같은 조건의 기준값인 것처럼 비교하지 않는다. 결합 CSV는 기존 36개와 신규 15개의 총 51행이며 `measurement_source`로 재사용/신규를 구분한다.

원본은 변경하지 않고 변환을 메모리에서 만든다. float32 RGB 0~255 입력, 얼굴 중앙 정사각형 자르기, 현미경 전체 리사이즈와 PMG 네 logit 합산을 유지한다. 모델·코드·프로필·분할 해시가 달라지면 중단한다.

## 모델 선정과 해석

완료한 4차 Validation에서 기본·2차·4차를 비교한 후속 후보인 **피부 4차·얼굴 4차·두피 2차**를 미리 고정했다. 기본~3차는 기존 Test 결과로 비교한다. 다섯 모델 모두를 새 4차 Validation 조건에서 선정한 결과라고 표현하지 않는다. 공식 배포 후보 색인은 변경하지 않는다.

Accuracy·Macro-F1·클래스별 Recall·혼동행렬과 기본 대비 변화를 기록한다. 저장한 사진별 예측 점수는 검증된 정답 확률이 아니다. 실제 촬영 4명·124장에는 질환 정답이 없어 실제 장비 Target Accuracy·F1을 계산하지 않는다. 과거 저장 모델과 현재 Test의 독립성, 사람·병변·세션 수준 누수는 완전히 확인되지 않았다. 단일 seed의 후속 Source 평가로 해석한다.

## 실행과 결과

Colab에서 새 Test 노트북을 열어 **GPU 선택 → 모두 실행 → Drive 연결**한다. 별도 실행 ZIP 업로드·실제 사진 복호화 키·새 학습은 필요 없다. 기존 `mediflow_Project/MediFlow_Adaptation_Common_v3.zip`, 세 원본 데이터 ZIP, 1차 분할 목록과 완료한 4차 모델을 사용한다. 기본~3차 점수는 노트북에 포함돼 있어 Drive의 이전 Test 폴더나 2·3차 모델 폴더를 새로 읽을 필요가 없다.

4차 입력은 기존 Drive의 `MediFlow_adaptation_runs/4ae6e07eb4af1225/measured_strength_492ee01afb45`다. 실제로 존재하는지 실행 시작에 검사하며 다른 설정 폴더로 임의 대체하지 않는다. 새 결과는 같은 상위 폴더의 `stage4_test_<평가 계획 해시 앞 12자리>`에 보관한다. 이전 Test 결과 폴더를 덮어쓰지 않는다.

결과 ZIP 이름은 `MediFlow_Stage4_Test_Results_20261111.zip`이다. Drive의 새 결과 폴더 옆에 보관하고 완료 후 다운로드한다. 모델과 원본 이미지는 다시 넣지 않는다.

- `test_comparison.csv`: 기존 36개와 신규 15개를 합친 51개 비교의 Accuracy·Macro-F1. 같은 조건의 기본 점수가 있을 때만 변화량을 기록한다.
- `class_metrics.csv`, 분야별 JSON: 클래스별 Precision·Recall·F1·혼동행렬.
- `plots`: 기존 결과와 신규 4차의 비교 그래프, 새로 평가한 4차의 원본·기존 확장·4차 확장 혼동행렬. 이전 모델의 미측정 조건에는 선을 그리지 않는다.
- 분야별 `predictions.npz`와 `test_rows.json`: 동일 사진 순서의 예측 점수·정답·경로. NPZ는 `allow_pickle=False`로 읽을 수 있다.
- `evaluation_plan.json`, `summary.json`, `code`, `logs`: 고정 계획·완료 상태·실행 코드·환경·오류 로그.
- `previous_test`: 재사용한 이전 평가 계획·완료 기록·분야별 결과 JSON 원본. 새 분야별 JSON과 예측 파일은 4차의 신규 추론만 담는다.

같은 노트북을 다시 실행하면 계획·코드·예측 파일 해시가 같은 완료 분야를 재사용한다. 진행 장수와 실제 오류를 화면에 표시한다. `summary.json`의 세 분야 완료, `comparisons=15`, `reused_comparisons=36`, `combined_comparisons=51`을 확인한 뒤 완료로 판정한다.

[평가 코드](../../../src/mediflow_datasets/adaptation_stage4_test.py) · [생성기](../../../scripts/build_adaptation_stage4_test.py) · [검사](../../../tests/test_adaptation_stage4_test.py).

## 로컬 확인 결과

4차만 추론하는 수정본의 Ruff 검사를 통과했고 전체 pytest는 **199개 통과·3개 건너뜀**이다. 건너뛴 항목은 기본 로컬 환경의 matplotlib 미설치에 따른 기존 그래프 검사다. 관련 8개 검사는 모두 통과했다.

실제 저장된 4차 모델 3개를 각각 로드해 합성 이미지 두 장과 다섯 변환으로 추론했다. 기존 모델은 새로 로드하지 않는 경로와 이전 결과 해시 검증을 확인했다. 별도 준비된 matplotlib으로 합성 4차 점수의 비교 그래프·혼동행렬을 생성하고 확인했다. 이는 기능 검사이며 실제 Source Test Accuracy를 새로 측정한 결과가 아니다. 노트북에 포함한 평가 계획·실행 코드·기존 결과 바이트도 현재 로컬 파일과 일치했다.

## Colab 코드 사본 불일치 수정

사용자가 전달한 실제 Colab 로그에서 TensorFlow·Keras 설치와 GPU 인식은 성공했다. 이후 `Evaluation code changed: adaptation_final_test.py`가 발생해 Test 추론 전에 중단됐다. 해당 실행의 결과 폴더 식별값은 `stage4_test_f1e59b39b583`이며 새 Test 완료 결과로 세지 않는다.

원인은 생성기가 혼동행렬 그림 파일명을 전체 문자열에서 바꾸면서 노트북에 포함한 공통 평가 코드 사본의 같은 문자열까지 바꾼 것이다. 원본과 포함된 사본을 비교해 이 한 줄의 차이를 확인했다. 파일명 변경을 Colab의 그림 표시 구문에만 한정하고 공통 코드를 원본 바이트로 유지했다. 기존 코드 해시 검사는 그대로 둔다.

환경을 설치하기 전에 준비된 실행 코드 5개의 해시를 검사하고, 노트북 생성 때도 포함한 코드 바이트를 확인하도록 보완했다. 새 검사는 노트북이 실제로 ZIP을 풀고 코드를 쓰는 과정까지 실행해 원본 일치를 확인한다. 수정 전 같은 오류를 재현했고 수정 후 통과했다. 실제 공통 ZIP v3로도 준비된 코드 5개·이전 결과 5개·평가 계획 바이트를 확인했다. 이 검사는 외부 Python 실행·설치·GPU 추론 직전에 중단하는 로컬 점검이며 실제 Colab Test를 대신하지 않는다.

모델·Test 목록·변환·seed·평가 조건은 바꾸지 않았다. 생성기 식별값이 달라져 수정본 결과 폴더는 `stage4_test_dda96d8dc716`으로 구분된다. Drive의 이전 오류 로그는 삭제하지 않는다. 첫 출력의 `MediFlow 4차만 Test · 코드 사본 검증 수정본`과 `평가 코드 사본 검사 완료`를 확인한다. 현재 로컬 노트북을 Colab에서 다시 열고 실행하며, 이미 열어 둔 이전 사본에는 수정이 자동 반영되지 않는다.

수정 후 관련 9개 검사는 모두 통과했다. Ruff와 전체 pytest는 **200개 통과·3개 건너뜀**이며 건너뛴 항목은 기본 로컬 환경의 matplotlib 부재에 따른 기존 그래프 검사다. 저장 모델 15개의 해시, 이전 Test 결과와 완료한 4차 설정의 해시도 변경되지 않았다. 이후 실제 Colab Test·결과 회수를 완료했고 [결과 분석](ADAPTATION_STAGE4_TEST_RESULT_REVIEW_20261111.md)에 기록했다.
