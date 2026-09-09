# MediFlow Google Drive 정리안

2026-09-08. Google Drive 연결로 관련 폴더 목록을 실제 확인했다. 이번 작업은 목록 확인과 정리안 작성이며 Drive 파일 이동·삭제·폴더 생성·업로드는 수행하지 않았다.

## 권장 방향

기존 데이터·실험·모델 경로는 유지하고 노트북 분류와 연구 문서 보관부터 정리한다. 학습·재개·패키징 노트북에는 Drive의 현재 경로가 저장되어 있다. 상위 폴더를 통째로 MediFlow라는 새 폴더로 옮기려면 해당 경로를 사용하는 실행 코드도 함께 정비해야 한다.

| 현재 폴더 | 실제 확인 내용 | 제안 |
|---|---|---|
| [mediflow_datasets](https://drive.google.com/drive/folders/1YZOwAUrEQeSZxP0cI0Kz8pna5ztJZxTB) | Hair Clean ZIP 및 검사 보고서 폴더 | 위치 유지 |
| [mediflow_notebooks](https://drive.google.com/drive/folders/1zFPNupqGtCohI8NCAyr8Q2MwVZ76zXu9) | Hair 10개, Web Skin 4개 | hair / web_skin / archive로 분류 |
| [mediflow_experiments](https://drive.google.com/drive/folders/1THxR-kebRZ9Am7AOWnRNwE2nsNKn6vZ9) | hair / web_skin으로 이미 분리 | 기존 실험 이름과 폴더 유지 |
| [mediflow_models](https://drive.google.com/drive/folders/1RxPoGkdcyuxVnO98D1p02zG1kEsTsqFu) | 두 후보의 패키지 폴더와 ZIP, Web Skin 해시 파일 | 현재 구조 유지 |
| [자율설계2](https://drive.google.com/drive/folders/1JJyjQ9dvozWOim_VVNDRWk9wyrpOA9UG) | 데이터 ZIP 3개와 과거 결과 ZIP 3개 | 원본 보관. 현재 사용하는 web_skin_processed.zip 위치 유지 |
| mediflow_reports (신규 제안) | 현재 루트 목록에 없음 | hair / web_skin / common으로 연구 문서 보관 |

자율설계_eye_project와 software는 이번 MediFlow 피부·두피 정리 범위에 포함하지 않았다. 하위 내용도 열람하지 않았다.

## 노트북별 제안

- `hair/`: Clean 데이터 생성, 224·256 학습, 오답 감사, LS·Focal·B1·추가 5 Epoch, 후보 패키징 노트북 9개.
- `web_skin/`: 데이터 감사, `web_skin_all_experiments_reviewed_colab.ipynb`, 후보 패키징 3개.
- `archive/`: `hair_efficientnetB0_partial_finetuning_colab.ipynb`와 `web_skin_all_experiments_colab.ipynb` 2개.

archive는 삭제 대상이 아니라 과거 시도/검토 전 버전의 보관 위치다. 학습 출력이 들어 있는 노트북을 빈 로컬 사본으로 덮어쓰지 않는다. 이동할 때는 파일 ID와 현재 부모 폴더를 다시 확인하고, 이동 후 목록을 검증한다.

## 문서 폴더 제안

- `hair/`: Hair 실험 총정리, 기존 고급 기법 보류 기록.
- `web_skin/`: Web Skin 전체 총정리, 후보 검증 완료 기록.
- `common/`: 두 모델 실험 원리·효과 해설, 후속 개선 계획, 이 정리안.

이 문서들은 현재 로컬 `docs/research/`에 있다. 아직 Drive 업로드를 수행한 것은 아니다.

## 폴더와 ZIP이 함께 있는 이유

실험 폴더는 학습 재개·개별 파일 확인·패키징 입력에, reports ZIP은 그래프·수치 공유에, full_models ZIP은 전체 체크포인트 백업에 사용한다. 후보 폴더와 후보 ZIP 역시 실행 파일 확인과 배포·보관 역할이 다르다.

Web Skin full_models ZIP은 Drive 메타데이터상 617549206 bytes이며 reports ZIP은 24869997 bytes다. 파일명이나 크기만 보고 불필요한 중복이라고 판단하지 않는다. 이번 목록 확인은 압축 내부와 원본 폴더의 완전 일치 검사가 아니다.

용량 확보가 필요해지면 원본 폴더와 백업 ZIP의 내용·해시, 복원 가능 여부 및 재개에 필요한 체크포인트를 확인한 뒤 별도로 삭제 범위를 결정한다. 현재 정리안에는 삭제 작업이 없다.

## 진행 순서

1. 기존 경로를 유지한 채 노트북 14개를 위 분류로 정리한다.
2. mediflow_reports를 만들고 최신 연구 문서를 분야별로 보관한다.
3. 자주 여는 모델·데이터·노트북 폴더를 즐겨찾기로 표시한다.
4. 상위 폴더 통합이나 대용량 파일 삭제는 코드 경로·복원 검증을 포함한 별도 작업으로 진행한다.

현재 완료: 관련 폴더 목록 확인 및 구체적인 분류안 작성. 미완료: 실제 Drive 변경. Colab 경로가 영향을 받는 데이터나 결과 이동을 이 정리안만으로 완료했다고 간주하지 않는다.
