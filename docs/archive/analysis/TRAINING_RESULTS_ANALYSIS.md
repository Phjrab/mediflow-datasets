# Training Results Analysis

이 문서는 현재 저장소에 있는 학습 산출물을 기준으로 성능을 정리하고,
다음 학습에서 우선적으로 바꿔야 할 방향을 제안합니다.

## 1) 실험 요약

- Backbone: EfficientNet-B0
- Input size: 224x224
- Epoch: 15
- 비교 방식: Original vs Augmented

## 2) 모델별 핵심 지표

### 2-1. Skin Microscopy (10-class)

| Variant | Best Val Acc | Test Acc | Best Epoch |
|---|---:|---:|---:|
| Original | 0.9950 | 0.9943 | 13 |
| Augmented | 0.9950 | 0.9957 | 14 |

해석:
- 이미 매우 높은 성능(99%+)이며 Augmented가 소폭 우세합니다.
- 여기서는 정확도 개선보다 도메인 일반화 검증(실장비 데이터) 중요도가 더 큽니다.

### 2-2. Web Skin (5-class)

| Variant | Best Val Acc | Test Acc | Macro Precision | Macro Recall | Macro F1 |
|---|---:|---:|---:|---:|---:|
| Original | 0.6680 | 0.7850 | 0.7979 | 0.7850 | 0.7833 |
| Augmented | 0.6880 | 0.8200 | 0.8271 | 0.8200 | 0.8196 |

해석:
- Augmented가 전 지표에서 개선되었습니다.
- 특히 Test Acc가 +3.5%p(0.7850 -> 0.8200) 상승했습니다.

Original 클래스별 F1:
- 건선 0.7750
- 아토피 0.7218
- 여드름 0.7039
- 정상 0.9581
- 주사 0.7578

Augmented 클래스별 F1:
- 건선 0.8000
- 아토피 0.7801
- 여드름 0.7471
- 정상 0.9756
- 주사 0.7950

포인트:
- 아토피/여드름 개선 폭이 크고, 정상 클래스는 원래도 강한데 더 상승했습니다.
- 여전히 아토피/여드름/주사 구간은 클래스 유사성 영향이 남아있습니다.

### 2-3. Hair Microscopy (5-class)

| Variant | Best Val Acc | Test Acc | Macro F1 | Best Epoch |
|---|---:|---:|---:|---:|
| Original | 0.7062 | 0.6841 | 0.6827 | 14 |
| Augmented | 0.7048 | 0.6910 | 0.6895 | 15 |

해석:
- Test Acc +0.69%p, Macro F1 +0.0068로 소폭 개선되었습니다.
- Web Skin 대비 개선폭이 작아, 단순 증강만으로는 한계가 보입니다.

Augmented 클래스별 F1:
- 모낭사이홍반 0.7352
- 미세각질 0.6007
- 비듬 0.6562
- 탈모 0.7700
- 피지과다 0.6853

포인트:
- 미세각질이 최저 F1(0.6007)로 병목 클래스입니다.
- 비듬/피지과다도 상대적으로 낮아, 텍스처 유사 클래스 분리 강화가 필요합니다.

## 3) 지금 바꿔야 할 학습 방향 (우선순위)

### 우선순위 A: Fine-tuning 단계 분리

권장:
1. Head 학습(백본 freeze)
2. 마지막 block 일부 unfreeze
3. 낮은 LR로 재학습

초기 제안값:
- Stage 1 LR: 1e-4
- Stage 2 LR: 1e-5
- Epoch: 10 + 10 (또는 15 + 10)

기대효과:
- 도메인 특화 특징 적응 향상
- 특히 Hair의 미세 텍스처 분리 성능 개선 가능성

### 우선순위 B: 입력 해상도 비교 (Hair 우선)

권장 실험:
- 224 -> 256 비교
- 필요 시 300까지 확장

이유:
- 미세각질/비듬/피지과다는 작은 패턴 차이가 중요함
- 저해상도에서 핵심 텍스처가 소실될 가능성 큼

### 우선순위 C: 손실함수/학습전략 개선

권장:
- Focal Loss (gamma 1.5~2.0 범위 탐색)
- Label Smoothing (0.05~0.1)

이유:
- 클래스 불균형보다 클래스 간 분리도 문제(class separability)가 현재 핵심
- 애매한 샘플에 대한 decision boundary 안정화에 도움

### 우선순위 D: 클래스별 하드케이스 수집/검증

권장:
- 미세각질 vs 비듬 vs 피지과다 혼동 샘플 별도 폴더 수집
- 예측 확률 상위 2개 차이(Top1-Top2 margin)가 작은 샘플 수동 검토

이유:
- 데이터 양보다 라벨 명확성/경계 품질이 성능을 더 좌우하는 단계

## 4) 모델별 운영 권장안

- 배포 기본값:
	- Skin: augmented
	- Web Skin: augmented
	- Hair: augmented

- 운영 경고:
	- 장비가 다르면(웹캠 vs 현미경) 모델을 절대 혼용하지 않기
	- 실장비 데이터셋으로 별도 검증 세트 구축 전까지, 공개 데이터 성능을 그대로 현장 성능으로 해석하지 않기

## 5) 다음 실험 플랜 (2주 스프린트 예시)

Week 1:
1. Hair 256 해상도 + 2-stage fine-tuning
2. Hair Focal Loss on/off A/B 테스트
3. 미세각질 관련 오분류 수집 리포트 작성

Week 2:
1. Web Skin 동일 전략(2-stage fine-tuning)
2. Top-k confusion 비교표 작성
3. 실장비 소규모 검증셋(클래스당 n>=30) 파일 분리

완료 기준:
- Hair Macro F1 >= 0.72 또는 미세각질 F1 >= 0.66
- Web Skin Macro F1 >= 0.83

## 6) 결론

- Skin 10-class는 이미 높은 성능으로 안정 구간입니다.
- Web Skin은 Augmented 효과가 명확하며, 추가 fine-tuning으로 더 끌어올릴 여지가 있습니다.
- Hair는 현재 병목이 뚜렷하므로 해상도/미세조정/손실함수 개선을 우선 적용하는 것이 가장 효율적입니다.

