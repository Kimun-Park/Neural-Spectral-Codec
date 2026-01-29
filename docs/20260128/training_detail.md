# Training Strategy 상세 설계

> 이 문서는 [overall_approach.md](overall_approach.md)의 학습 전략 및 평가 방법론의 상세 내용을 다룹니다.

---

## 1. 개요

### 1.1 목적

GNN 모델을 **metric learning** 방식으로 학습하여, 같은 장소는 가깝게, 다른 장소는 멀게 embedding합니다.

### 1.2 설계 원칙

| 원칙 | 설명 |
|-----|------|
| Metric Learning | Triplet loss로 embedding 공간 학습 |
| Per-Sequence Mining | 시퀀스 간 triplet 혼합 방지 |
| Multi-Dataset | 다양한 센서/환경 데이터 혼합 학습 |

---

## 2. Triplet Loss

### 2.1 정의

$$\mathcal{L}_{\text{triplet}} = \sum_{(a, p, n)} \max(0, \lVert z_a - z_p \rVert_2 - \lVert z_a - z_n \rVert_2 + m)$$

| 기호 | 설명 |
|-----|------|
| $z_a$ | Anchor embedding |
| $z_p$ | Positive embedding (same place, different time) |
| $z_n$ | Negative embedding (different place) |
| $m$ | Margin (기본: 0.1) |

### 2.2 Pair 정의

| 유형 | 조건 |
|-----|------|
| Positive | 거리 < 5m, temporal gap > 30 frames |
| Negative | 거리 > 30m |

### 2.3 Margin 선택

| Margin | 효과 |
|--------|------|
| 0.05 | 느슨한 제약 |
| **0.1** | 표준 설정 |
| 0.2 | 강한 제약 |

---

## 3. Triplet Mining

### 3.1 Per-Sequence Mining

서로 다른 sequence 간 triplet 생성 금지

**이유:**
- 다른 sequence 장소를 "다른 장소"로 잘못 정의하면 false negative 발생
- 좌표계가 sequence마다 다를 수 있음

### 3.2 Semi-hard Mining

$$d(a, p) < d(a, n) < d(a, p) + m$$

Positive보다 멀지만 margin 이내인 negative 선택 → 가장 informative한 학습 신호

### 3.3 Temporal Constraint

연속 프레임은 거의 동일 → positive로 사용하면 trivial

**해결:** 최소 30 frames (3초 @ 10Hz) temporal gap 강제

---

## 4. Multi-Dataset Training

### 4.1 데이터셋 구성

| Dataset | 센서 | 환경 | 역할 |
|---------|------|------|------|
| KITTI | HDL-64E | 도심/고속도로 | Train/Val/Test |
| NCLT | Velodyne-32 | 캠퍼스 | Train |
| HeLiPR | VLP-16 | 다양한 환경 | Train |

---

## 5. Evaluation

### 5.1 기존 평가 방식의 문제점

초기 구현에서는 **Loop Closure pair 간의 descriptor 유사도**를 평가 메트릭으로 사용했습니다. 이는 잘못된 설정이었습니다.

| 방식 | 문제점 |
|-----|--------|
| Pair 유사도 | Loop closure를 이미 알고 있다고 가정 |
| | Retrieval 능력을 평가하지 못함 |

**수정:** Query에 대해 전체 database에서 retrieval한 뒤 R@K 평가

### 5.2 Loop Closure 기반 평가

| 방식 | 특징 |
|-----|------|
| All-vs-all | 낙관적 결과 |
| **Loop closure** | 실제 revisit 시나리오 → 현실적 결과 |

### 5.2 Query 정의

Loop closure query: 이전에 방문한 장소를 다시 방문하는 프레임

조건:
- 과거 프레임과 거리 < 5m
- 과거 프레임과 temporal gap > 30 frames

### 5.3 평가 Metrics

| Metric | 설명 |
|--------|------|
| **R@1** | Top-1에 정답 포함 비율 (가장 중요) |
| R@5 | Top-5에 정답 포함 비율 |
| R@10 | Top-10에 정답 포함 비율 |

---

## 6. Cross-Sensor Generalization

### 6.1 센서 차이

| 특성 | VLP-16 | HDL-64E |
|-----|--------|---------|
| Channels | 16 | 64 |
| Vertical FOV | 30° | 26.8° |
| Points/scan | ~30K | ~120K |

### 6.2 정규화 전략

학습과 평가 모두 `target_elevation_bins=16`으로 통일

---

## 7. 실험 결과

### 7.1 학습

| 항목 | 값 |
|-----|-----|
| Total training time | 9.73 hours |
| Best validation R@1 | 1.0000 |
| Early stopping | Epoch 15/50 |

### 7.2 KITTI 평가

| Sequence | Queries | GNN R@1 | Raw R@1 | GNN 효과 |
|----------|---------|---------|---------|---------|
| KITTI 09 | 18 | 61.11% | 77.78% | -16.67% |
| KITTI 00 | 833 | 62.79% | 60.14% | +2.65% |

---

## 8. 요약

| 구성요소 | 역할 |
|---------|------|
| **Triplet Loss** | Metric learning으로 embedding 학습 |
| **Per-Sequence Mining** | 시퀀스 간 false negative 방지 |
| **Semi-hard Mining** | Informative 학습 신호 |
| **Multi-Dataset** | 센서/환경 일반화 |
| **Loop Closure Evaluation** | 실제 시나리오 기반 평가 |

---

**문서 생성일:** 2026-01-28
