# Neural Spectral Histogram Codec: Overall Approach

---

## 0. 핵심 직관

### 문제 (Why)

자율주행 및 로봇 SLAM에서 **Loop Closure Detection**은 누적 오차를 보정하기 위한 핵심 기술입니다.

| 과제 | 설명 |
|-----|------|
| 회전 불변성 | 같은 장소에서 다른 방향으로 관측해도 인식 |
| 센서 일반화 | VLP-16 → HDL-64E 등 다른 센서에도 동작 |
| 계산 효율성 | 실시간 처리를 위한 효율적 retrieval |

### 기존 방법의 한계

| 방식 | 장점 | 단점 |
|------|------|------|
| PointNet 계열 | 표현력 높음 | 회전에 민감 |
| Hand-crafted | 회전 불변 | 표현력 제한 |
| BEV 기반 | 2D 처리 가능 | 높이 정보 손실 |

### 목표 (Goal)

| 목표 | 구현 방법 |
|-----|----------|
| 회전 불변 표현 | FFT magnitude |
| 센서 독립 정규화 | target_elevation_bins |
| 높이 정보 보존 | Per-elevation histogram (800D) |
| 시간적 맥락 주입 | GNN message passing |

---

## 1. Pipeline Stages

```
LiDAR Point Cloud
       ↓
[Stage 1] Range Image Projection
       ↓
[Stage 2] Row-wise FFT + Magnitude
       ↓
[Stage 3] Per-Elevation Spectral Histogram (800D)
       ↓
[Stage 4] Keyframe Selection
       ↓
[Stage 5] Temporal Graph Construction
       ↓
[Stage 6] GNN Enhancement (3-layer GAT)
       ↓
[Stage 7] Loop Closure Retrieval
```

| Stage | 입력 | 출력 | 핵심 |
|-------|------|------|------|
| 1. Range Image | Point Cloud | (H, W) image | Spherical projection |
| 2. FFT | Range Image | Magnitude | Row-wise FFT |
| 3. Histogram | Magnitudes | 800D vector | Per-elevation binning |
| 4. Keyframe | Scans + poses | Keyframes | 4-criterion selection |
| 5. Graph | Keyframes | G = (V, E) | k-temporal neighbors |
| 6. GNN | Graph | Enhanced embeddings | 3-layer GAT |
| 7. Retrieval | Query, Database | Top-K candidates | Cosine similarity |

---

## 2. 핵심 아이디어

### 2.1 Rotation Invariance via FFT Magnitude

같은 장소를 다른 방향에서 관측 → azimuth 방향 cyclic shift 발생

**FFT shift theorem:** Magnitude는 shift에 불변

### 2.2 Sensor-Agnostic Normalization

| 센서 | Native Rings | 정규화 후 |
|------|-------------|----------|
| VLP-16 | 16 | 16 bins |
| HDL-64E | 64 | 16 bins |
| OS-128 | 128 | 16 bins |

### 2.3 GNN Context Injection

독립적인 descriptor만으로는 perceptual aliasing 해결 어려움

**해결:** Temporal trajectory context를 GNN으로 주입

---

## 3. Requirements

| 요구사항 | 구현 |
|---------|------|
| R1. Rotation Invariance | FFT magnitude |
| R2. Sensor Generalization | target_elevation_bins |
| R3. Computational Efficiency | 800D descriptor |
| R4. Discriminative Power | GNN + triplet learning |
| R5. Temporal Consistency | Temporal graph |

---

## 4. 현재 진행 상황

### 완료

- [x] Range Image projection
- [x] FFT + Spectral Histogram 인코더
- [x] Keyframe Selection (4-criterion)
- [x] GNN 모델 (3-layer GAT)
- [x] Multi-dataset 학습 파이프라인
- [x] HeLiPR 데이터셋 학습 (19개 시퀀스)

### 진행 중

- [ ] Cross-sensor 일반화 평가

### 실험 결과

| Dataset | Queries | GNN R@1 | Raw R@1 |
|---------|---------|---------|---------|
| KITTI 09 | 18 | 61.11% | 77.78% |
| KITTI 00 | 833 | 62.79% | 60.14% |

---

## 5. Perceptual Aliasing 분석

### 핵심 질문

멀리 떨어진 두 장소에서 descriptor가 비슷할 때, 실제로도 비슷한가?

### 실험 결과 (KITTI 00)

| 유형 | 거리 | 유사도 | 분석 |
|------|------|--------|------|
| True Negative | 248m | 0.898 | Range Image 확연히 다름 |
| False Positive | 174m | 0.996 | Range Image 실제로 비슷 |
| True Positive | 0.5m | 0.999 | Loop Closure |

### 핵심 발견

| 질문 | 답변 |
|------|------|
| Descriptor가 비슷하면 실제로 비슷한가? | **Yes** |
| Descriptor 문제인가? | **No** - Legitimate Aliasing |
| GNN이 왜 필요한가? | Trajectory 기반 구분 |

---

## 6. 관련 문서

| 문서 | 내용 |
|------|------|
| spectral_encoding_detail.md | Range Image + FFT + Histogram |
| gnn_detail.md | GNN 아키텍처 |
| keyframe_detail.md | Keyframe Selection |
| training_detail.md | 학습 전략 |
| perceptual_aliasing_analysis.md | Aliasing 분석 |

---

**문서 생성일:** 2026-01-28
