# 팀원 LLM VLM 저장소 분석과 프로젝트 연결 방향

> 2026-11-11 정리 상태: **검토·설계 기록, MediFlow LLM/VLM·장비 통합은 미완료**. [현재 작업 안내](../../README.md) · [전체 문서 대조표](../../DOCUMENT_INDEX.md). 아래 본문은 작성 시점 기록이며, 본문·원본 JSON의 이전 경로는 정리 대조표로 추적한다.

기준일 2026-10-26. 두 저장소의 기본 브랜치와 개발 브랜치, 실제 학습 코드, 공개 평가 기록,
데이터 처리, 실행 환경, GitHub Actions를 대조했다. 기존 피부·두피 프로젝트의 모델 계약은
[선행 종합 점검](PRE_LLM_VLM_PROJECT_AUDIT_20261026.md)을 기준으로 연결했다.

## 결론과 바로 다음 작업

**팀원의 최근 학습 성과를 재사용하면서 우리 프로젝트용 과제를 추가하는 방향이 맞다.**
핵심 연구 코드는 MedVLM의 최신 개발 브랜치를 사용하고 Forge는 선택적인 학습 관리 도구로 둔다.
현재 의료 문답과 방사선 영상 VQA 결과를 피부·두피 성능으로 해석하지 않는다.
추가 학습에 앞서 저장된 분류 결과를 설명하는 LLM부터 실행하고, 별도 경로에서 사진을 직접
입력받는 VLM의 질환 분류를 평가한다.

사용자가 원하는 두 흐름은 다음과 같다.

1. **웹캠 또는 현미경 사진 → 기존 전문 분류 모델 → 분류 결과와 문진 → 설명 LLM.**
   LLM은 어떤 분류 결과가 나왔는지와 문진을 연결해 한국어로 설명한다.
2. **사진과 촬영 도메인 → 독립 VLM → 질환 분류 후보와 시각 근거.**
   이 경로는 기존 분류기의 라벨·점수를 입력받지 않고 별도 판별한다.

이것을 질환 진단 연구의 두 경로로 개발하되, 사진의 분류 성능과 사람의 진단 타당성을
구분해 평가한다. 실제 시스템 연결은 이 두 기능의 검증 뒤에 진행할 수 있다.

## 1 검토한 실제 버전

| 저장소 | 기준 브랜치와 고정 커밋 | 확인한 상태 |
|---|---|---|
| mediflow-datasets | 로컬 기준 `ffd71eca68ff1e21c145ed56158f4cfa77f7f037`와 현재 변경분 | 기존 분류 모델·평가·전처리·자료의 기준 |
| mediflow-medvlm 기본 | main `87df64d03acb79eb86f432456141aa9a518239b5` | 초기 요청 검증과 워크플로 골격 |
| mediflow-medvlm 최신 연구 | codex/medical-finetune-v1-20261023 `e16f8c6df3244a78cd9c30ba5ce7e43d5b8f45a3` | 실제 데이터·학습·재로드·평가 코드와 공개 결과 |
| forge-finetune-dashboard | main `d9a4656a5bee38cc9903e8caf5c4b680709423d2` | 실제 CUDA 학습 워커와 웹 관리 화면 |
| Forge 후속 문서 | codex/ui-forge-training-workbench `74e98a44279248906b77f87cbe5bca2d0add18cb` | main 코드의 배포 확인 문서 추가 |

MedVLM 최신 연구 브랜치는 main보다 5개 커밋 앞서 있으며 조회 시 PR 목록은 비어 있었다.
기본 clone 후 main만 읽으면 최근 학습 기능과 결과가 보이지 않는다.
Forge 후속 브랜치와 main의 비교에서는 배포 문서 3개만 달랐고 학습 코드 변경은 없었다.
따라서 Forge의 학습 검토는 main 기준으로 유효하다.

고정 커밋의 원격 CI는 [MedVLM Test](https://github.com/Phjrab/mediflow-medvlm/actions/runs/39965707522),
[Forge Test](https://github.com/Phjrab/forge-finetune-dashboard/actions/runs/37605147909)에서
성공 상태를 확인했다. 이는 코드 검사 결과이며 GPU 학습이나 의료 품질 평가의 대체 자료는 아니다.

## 2 MedVLM에 실제로 구현된 기능

### 초기 main과 최신 브랜치의 차이

초기 main에는 VQA·보고서·시계열 비교 요청과 응답 검증만 있고 실제 모델 로더는 없었다.
최신 브랜치에는 이미지 없는 TEXT_QA, 데이터 준비, Transformers와 PEFT 학습, 고정 평가,
결과 집계까지 추가됐다. 초기 main의 Python 3.10 StrEnum 불일치와 문자열 task 검증 문제는
최신 브랜치에서 수정됐다. 이 두 문제를 최신 코드의 오류로 다시 나열하지 않는다.

### 재사용 가치가 큰 부분

원본 출처·revision·원본 ID와 모델 입력·정답을 분리하는 canonical 데이터 구조가 있다.
문서/이미지/중복 그룹을 검사하고 고정 분할의 내용 해시를 만든다. 학습용 메시지를 만들 때
질문·문맥·선택지·이미지만 허용하며 정답과 provenance를 사용자 입력에 넣지 않는다.

학습은 모델의 native chat template를 사용한다. 답변 토큰만 학습하도록 프롬프트·padding·
이미지 토큰을 마스킹하고, 실제 프롬프트 토큰 경계가 일치하는지 검사한다. Vision encoder와
projector를 기본 동결한 뒤 언어 모듈에 LoRA를 적용한다. NF4 QLoRA, 정확한 모델 revision,
optimizer/RNG 상태, 코드·데이터 해시를 포함하는 재개와 새 프로세스 재로드를 구현했다.

Validation loss로 체크포인트를 선정하고, 고정된 평가 문항에서 base와 tuned를 비교한다.
실패도 평가 분모에 남기며, 이미지 포함/제외 비교와 그룹 단위 bootstrap도 있다.
이 부분은 단순히 학습 그래프를 보여주는 도구보다 연구 재현에 유용하다.

주요 코드 근거는 [학습 로더](https://github.com/Phjrab/mediflow-medvlm/blob/e16f8c6df3244a78cd9c30ba5ce7e43d5b8f45a3/src/mediflow_medvlm/training/models.py),
[입력 렌더링과 마스킹](https://github.com/Phjrab/mediflow-medvlm/blob/e16f8c6df3244a78cd9c30ba5ce7e43d5b8f45a3/src/mediflow_medvlm/training/rendering.py),
[학습 실행기](https://github.com/Phjrab/mediflow-medvlm/blob/e16f8c6df3244a78cd9c30ba5ce7e43d5b8f45a3/src/mediflow_medvlm/training/runner.py),
[고정 평가 실행기](https://github.com/Phjrab/mediflow-medvlm/blob/e16f8c6df3244a78cd9c30ba5ce7e43d5b8f45a3/src/mediflow_medvlm/evaluation/runner.py)다.

## 3 팀원의 실제 학습 결과와 해석

아래는 팀원이 공개 저장소에 보존한 실행 보고값이다. 이번 검토에서 팀원 PC의 원본 가중치를
다시 불러오거나 GPU 학습을 재실행한 것은 아니다. 공개 보고서에는 원본 예측의 해시와
base/tuned 연결 검사도 남아 있지만 원문 예측과 adapter 가중치 자체는 Git 밖에 있다.

| 모델 | 학습 자료와 완료 기록 | 현재 역할 후보 |
|---|---|---|
| Qwen3-1.7B | MedQuAD/PubMedQA, 525 updates, 처리 sample 8,400회, 재로드·paired 평가 완료 | 결과 설명 LLM의 가벼운 비교 후보 |
| Qwen3-4B | 같은 텍스트 과제, 525 updates, 처리 sample 8,400회, 재로드·paired 평가 완료 | 결과 설명 LLM의 첫 검토 후보 |
| Qwen3.5-2B | SLAKE VQA, 124 updates, 1,979 sample, 재로드·paired 평가 완료 | 일반 VLM 후보 |
| MedGemma 1.5 4B | SLAKE VQA, 124 updates, 1,979 sample, 재로드·paired 평가 완료 | 의료 VLM 후보 |

다른 fallback 모델은 준비 상태로 기록돼 있어 학습 완료 수에 넣지 않는다.
정제 canonical 19,866개는 모든 소스의 학습·검증·시험 자료를 포함한 수다.
19,866개를 전부 학습했다는 뜻이 아니다. KorMedMCQA는 한국어 평가, VQA-RAD는 외부 평가에 사용했다.

### 관찰된 성능 변화

공개 CSV의 정답 건수로 비교하면 다음과 같다. 자료별 지표를 서로 합쳐 하나의 성능으로 만들지 않는다.

| 모델과 평가 | Base | Tuned | 해석 |
|---|---:|---:|---|
| Qwen3-1.7B PubMedQA | 209/500 | 372/500 | 주어진 논문 문맥의 yes/no/maybe 답변 과제에서 상승 |
| Qwen3-4B PubMedQA | 177/500 | 384/500 | 같은 과제에서 상승. 한국어 분류 결과 설명 성능은 별도 |
| Qwen3.5-2B SLAKE | exact match 88/256 | exact match 185/256 | 방사선 영상의 짧은 정답 형식에서 상승 |
| MedGemma 1.5 SLAKE | exact match 0/256 | exact match 179/256 | 응답 길이와 출력 형식 영향이 커 의학 능력 상승폭으로 해석 불가 |
| Qwen3.5-2B VQA-RAD | exact match 19/64 | exact match 31/64 | 외부 방사선 VQA의 작은 표본에서 상승 |
| MedGemma 1.5 VQA-RAD | exact match 0/64 | exact match 30/64 | 동일한 형식·답 길이 한계를 함께 봐야 함 |

MedGemma base는 SLAKE에서 gold 1단어와 달리 긴 문장으로 답했고 생성 상한 32토큰에 도달한
사례가 있었다. 보고서의 0점은 엄격한 문자열 일치가 0이라는 뜻이다. 틀린 질환을 100% 골랐다는
뜻이 아니다. 반대로 tuned의 문자열 일치 상승만으로 한국어 피부·두피 진단을 입증하지도 않는다.

출처는 [실행 보고서](https://github.com/Phjrab/mediflow-medvlm/blob/e16f8c6df3244a78cd9c30ba5ce7e43d5b8f45a3/reports/public_summary/RESULTS_KO.md),
[원시 집계 CSV](https://github.com/Phjrab/mediflow-medvlm/blob/e16f8c6df3244a78cd9c30ba5ce7e43d5b8f45a3/reports/public_summary/paired_metrics.csv),
[모델 실행 카드](https://github.com/Phjrab/mediflow-medvlm/blob/e16f8c6df3244a78cd9c30ba5ce7e43d5b8f45a3/reports/public_summary/ADAPTER_CARDS.md)다.

### 우리 프로젝트와 아직 다른 점

LLM은 의료 질문에 답하도록 학습됐다. `candidate_id`, 도메인별 클래스 배열, 실제 분류 점수,
문진, 실행 실패를 받아 그대로 설명하는 과제는 평가하지 않았다. 한국어 의료 객관식 점수는
한국어 설명의 사실성이나 사용자 이해도와 다르다.

VLM 학습의 SLAKE와 외부 VQA-RAD는 CT·MRI·X-ray 중심이다. 얼굴 사진과 현미경 피부·두피는
다른 영상이다. 이미 학습한 VQA adapter가 이 세 입력에도 더 좋은지는 새 Validation에서 확인해야 한다.
MedGemma가 의료 이미지와 피부 이미지를 학습한 기반 모델이라는 점은 비교 이유가 되지만,
현미경 두피의 성능 보증은 아니다. [공식 모델 카드](https://developers.google.com/health-ai-developer-foundations/medgemma/model-card)

## 4 Forge의 역할과 제한

Forge는 실제로 CUDA LoRA/QLoRA를 수행하는 로컬 학습 작업대다. 데이터 검증·분할, Hub 모델
다운로드, 단일 GPU 큐, 일시정지·재개, 학습 손실, checkpoint ZIP을 구현했다. 합성 demo와
실제 GPU 경로도 구분한다. 따라서 화면만 있는 가짜 도구라고 평가하면 잘못이다.

하지만 전문 분류 결과를 받아 채팅하는 추론 서버는 없다. `adapter.py`는 학습 이벤트 연결 규격이며
MedVLM의 `generate()` 추론 어댑터와 다르다. 학습·내보내기 뒤 별도 추론 환경이 필요하다.

과거 GPU 검증은 Qwen 0.5B와 SmolVLM 256M의 합성 자료에서 각 10 optimizer updates를 실행한
파이프라인 확인이다. MedVLM의 10월 23일 실제 의료 학습은 Forge가 아니라 native 실행기로 진행됐다.
두 도구의 연결 근거는 텍스트 4행·7,603 bytes의 호환 예시 검증뿐이며 Forge로 전체 의료 데이터를
업로드·학습한 기록은 아니다.

근거는 [Forge README](https://github.com/Phjrab/forge-finetune-dashboard/blob/d9a4656a5bee38cc9903e8caf5c4b680709423d2/README.md),
[추론 endpoint 미지원 안내](https://github.com/Phjrab/forge-finetune-dashboard/blob/d9a4656a5bee38cc9903e8caf5c4b680709423d2/docs/adapter-contract.md),
[MedVLM의 Forge 호환 결과](https://github.com/Phjrab/mediflow-medvlm/blob/e16f8c6df3244a78cd9c30ba5ce7e43d5b8f45a3/reports/public_summary/data_forge_compatibility.json)다.

### 데이터와 학습 한도

Forge는 JSONL 128 KiB·1,000행, 한 행의 이미지 1장, 이미지별 2 MB·16 MP, 학습 snapshot
이미지 합계 128 MB 제한이 있다. 워커는 이미지를 최대 384×384 thumbnail로 먼저 줄인다.
context는 최대 2,048토큰이며 batch·rank·update 상한도 있다. 수만 장의 기존 Hair ZIP을
그대로 올리는 대규모 분류 학습 도구와는 사용 방식이 다르다.

Validation 첫 사례 하나의 생성 예시와 loss/perplexity는 제공하지만, 질환별 분류 정확도나
한국어 설명 평가를 대신하지 못한다. 최종 checkpoint가 가장 좋은 Validation checkpoint인지
자동으로 고르는 기능도 MedVLM native 실행기와 동일하지 않다.

## 5 코드에서 확인한 수정 필요 항목

아래는 고정 커밋에서 재현하거나 소스로 확인한 문제다. 이미 저장된 팀원 평가값이 모두
잘못됐다는 뜻은 아니다. 실행 경로와 사용 모델에 따라 영향을 구분한다.

| 우선순위 | 항목 | 근거와 실제 영향 | 권장 조치 |
|---|---|---|---|
| 연결 전 | 최신 연구 코드가 기본 main에 없음 | 최근 학습을 재현하려면 개발 브랜치 필요 | 사용할 커밋을 고정하고 PR 검토 뒤 release/tag 또는 main 반영 |
| 연결 전 | MedVLM 설명 LLM·질환 분류 VLM의 서비스 어댑터 없음 | `adapters.py:10–14`는 Protocol; 평가 실행기는 직접 `model.generate` 호출 | 기존 로더를 재사용한 실제 추론 어댑터와 작업별 입력 계약 추가 |
| 수정 필요 | MedVLM 응답 검증의 타입·공백·유한 수 검사 부족 | `workflows.py:52–68`; 숫자 answer, 빈 model ID, NaN latency가 통과함 | 엄격한 필드 타입·상태·domain 라벨 검사 |
| 연구 전 | Forge 분할은 행 단위 | `workspace.py:392–407`; 같은 이미지의 다른 질문이 Train/Val로 나뉠 수 있음 | 사진·사람·세션·파생 관계로 고정한 외부 split 지원 |
| 수정 필요 | Forge VLM 입력 형식 불일치 | `workspace.py:141–177`는 content image block 허용, `training.py:209`는 row images 강제 | 단일 정규형으로 변환; 당장은 문자열 content와 top-level images 사용 |
| 모델별 수정 | Forge template 파일 누락 | 다운로드 `huggingface.py:214`, export `training.py:460–480`에서 `.jinja` 제외 | 허용된 template/tokenizer 파일 보존과 export 후 재로드 검사 |
| 강화 권장 | MedVLM 평가 재개 identity에 실행 소스·환경 해시 미포함 | `evaluation/runner.py:183–204`는 버전 문자열·config 등을 비교 | 소스·라이브러리 digest도 identity에 포함하고 변경 시 새 run 생성 |
| 강화 권장 | raw 평가·compare 명령의 선정 확인이 campaign보다 약함 | raw evaluate는 adapter parent만 확인, compare는 명시적 base/tuned 역할 검사가 부족 | 새로운 workflow는 selected checkpoint·재로드·base/tuned 역할 gate를 공통 함수로 강제 |

Forge 이미지 입력 문제는 실제 `import → split → preflight`로 `KeyError('images')`를 확인했다.
동일 이미지에 서로 다른 질문을 둔 두 행도 Train/Val에 나뉘었다. template 누락은 임시 checkpoint에
`chat_template.jinja`를 넣고 실제 ZIP export를 호출해 제외되는 것을 확인했다.
기본 SmolVLM이 반드시 실패한다는 뜻은 아니며, 별도 template 파일을 요구하는 모델의 문제다.

MedVLM의 공식 `complete_v1.py` campaign은 선정·재로드 gate를 추가로 거친다. raw CLI의 약한
검사만으로 공개된 campaign 결과가 그 gate를 우회했다고 판단하지 않는다.

코드 위치:
[MedVLM 응답 검사](https://github.com/Phjrab/mediflow-medvlm/blob/e16f8c6df3244a78cd9c30ba5ce7e43d5b8f45a3/src/mediflow_medvlm/workflows.py#L52),
[Forge 입력 정규화와 분할](https://github.com/Phjrab/forge-finetune-dashboard/blob/d9a4656a5bee38cc9903e8caf5c4b680709423d2/workspace.py#L141),
[Forge 학습 사전 검사](https://github.com/Phjrab/forge-finetune-dashboard/blob/d9a4656a5bee38cc9903e8caf5c4b680709423d2/training.py#L209),
[Forge ZIP export](https://github.com/Phjrab/forge-finetune-dashboard/blob/d9a4656a5bee38cc9903e8caf5c4b680709423d2/training.py#L460).

## 6 실행 환경과 배포 판단

| 구분 | 현재 코드·기록 | 연결할 때의 판단 |
|---|---|---|
| 기존 전문 분류기 | TensorFlow 2.20.0·Keras 3.13.2 | 기존 RGB 0–255와 PMG 네 출력 계약 유지 |
| Forge 실제 학습 | PyTorch 2.8.0·Transformers 4.57.1·PEFT 0.17.1 | 최신 MedVLM 모델 지원을 이 버전으로 가정하지 않음 |
| MedVLM 최신 실험 | lock 기록의 PyTorch 2.14.1·Transformers 5.20.2·PEFT 0.21.2 | 팀원 실제 환경과 adapter revision을 묶어 재현 |
| 운영체제 | GPU lease의 `fcntl` 등 Linux 전제 | Windows 원생에서 그대로 학습·campaign 실행 불가 |
| 장비 | 보고된 실제 학습은 RTX 3060 12 GB | x86 CUDA 설치를 Jetson으로 그대로 옮기지 않음 |
| Colab | Linux GPU 환경이나 별도 지속성·중단 처리가 필요 | 학습 노트북으로 실행할 수 있게 구성하고 Forge 공개 서버 노출은 별도 검토 |

TensorFlow와 최신 PyTorch/PEFT 학습 환경을 한 가상환경에 강제로 맞추기보다, 분류기 실행과
언어·비전언어 모델 실행을 각각 관리하고 JSON으로 결과를 교환하는 방식이 적절하다.
구체적인 모델 메모리는 native profile의 실제 backward 검사로 확인하며 모델 이름만 보고
가능하다고 단정하지 않는다.

Forge 인증은 loopback 기본값, 초기 비밀번호 변경, 세션·CSRF·Origin·LAN view/control 구분을
구현하고 있었다. 확인된 인증 우회는 없었다. 토큰·훈련 원문·checkpoint는 Git 밖의 runtime
저장소에 두며, 기존 보호를 생략하고 임의 터널로 공개하는 설치 안내를 만들지 않는다.

## 7 이번 검증과 근거 수준

| 대상 | 이번 로컬 실행 결과 | 범위 |
|---|---|---|
| MedVLM 초기 main | 8개 통과 | FakeAdapter 기반 기본 계약 |
| MedVLM 최신 core·평가 subset | 64개 통과, Windows 권한 차이 1개 실패 | POSIX `0700` 검사 한계; 실제 GPU 추론 아님 |
| MedVLM 최신 학습 hardening subset | 25개 통과, 5개 skip, 3개 환경 오류 | Unix `fcntl` 미지원, 선택 GPU runtime 미설치 |
| Forge workspace | 33개 통과 | 데이터 정규화·준비 |
| Forge architecture | 2개 통과, 3개 skip | 선택 native runtime 미설치 |
| Forge training | 9개 통과, 2개 환경 오류 | Windows symlink 권한 제한 |
| Forge AuthStore | 11개 통과 | 비밀번호·세션 저장 검사 |
| Forge 전체 auth 파일 시도 | 12개 통과, HTTP 5개 환경 오류 | 샌드박스 loopback 연결 제한 |

subset은 일부 검사가 겹치므로 위 통과 수를 합산해 고유 테스트 수로 제시하지 않는다.
원격 Ubuntu CI 성공과 이번 Windows 환경의 제한을 함께 기록한다. 이번에 모델 다운로드,
추가 GPU 학습, 원격 서버 재시작, Git push 또는 PR 생성은 하지 않았다.

고정 커밋별 파일 목록·해시, 원격 CI/브랜치 상태, 로컬 검증 결과는
[점검 근거 폴더](../../../results/team_llm_vlm_review_20261026/README.md)에 보존했다.

## 8 프로젝트별 책임과 추천 진행 방향

`mediflow-datasets`는 기존 Skin·Web Skin·Hair 분류 모델과 원래 클래스 순서·전처리·평가를
관리한다. MedVLM은 새 설명 LLM·독립 질환 분류 VLM의 입력·추론·평가를 관리한다.
Forge는 지원 모델의 학습 관리 화면으로 사용하며, 의료 과제의 정답·데이터 독립성·성능 판단을
자동으로 대신하지 않는다. 기존 키오스크는 추후 촬영·문진·결과 화면을 연결하는 위치다.

바로 할 일은 네 모델을 다시 학습시키는 것이 아니다. 팀원 adapter와 기반 모델 revision을
확보한 뒤, 저장된 실제 분류 결과 몇 건을 넣어 Qwen3-4B의 base/medical-tuned 설명부터
비교한다. 원본 사진 전체를 준비하지 않아도 이 LLM 기능 실험은 가능하다.

VLM은 피부·얼굴·두피의 고정 Validation에서 MedGemma와 Qwen3.5의 base/VQA-tuned를
비교한다. 방사선 VQA adapter가 오히려 도메인 특화를 방해할 수도 있으므로 기반 모델을
반드시 비교군으로 둔다. 유망한 하나를 골라 도메인 자료로 추가 SFT할지는 그 결과로 정한다.

세부 입력·평가·파인튜닝·GitHub 작업 순서는
[LLM VLM 연결 실행 가이드](LLM_VLM_PROJECT_INTEGRATION_GUIDE_20261026.md)에 있다.
