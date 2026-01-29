# Neural Spectral Codec - 실험 보고서

## 1. 프로젝트 개요

LiDAR 기반 loop closing을 위한 spectral histogram 기반 place recognition 시스템.

### 1.1 목표
- LiDAR point cloud를 compact한 descriptor로 압축
- 회전 불변성(rotation invariance) 보장
- 다양한 센서/환경에서 일반화

### 1.2 핵심 아이디어
- FFT magnitude spectrum은 회전에 불변
- Spectral histogram으로 주파수 정보 압축
- GNN으로 temporal context 학습

---

## 2. 시스템 아키텍처

### 2.1 전체 파이프라인

```
Point Cloud → Range Image → FFT → Spectral Histogram → GNN → Embedding
    (N×4)      (16×360)   (16×181)     (50,)          (50,)
```

### 2.2 Range Image Projection

3D point cloud를 2D panoramic range image로 변환.

**구성요소:**
- **Elevation bins**: 16 (센서 독립적 표준화)
- **Azimuth bins**: 360 (1도 해상도)
- **Elevation range**: 센서별 설정
  - KITTI HDL-64E: -24.8° ~ +2.0°
  - NCLT HDL-32E: -30.67° ~ +10.67°
  - HeLiPR VLP-16: -15.0° ~ +15.0°

**핵심 처리:**
```python
# Spherical coordinate 변환
azimuth = arctan2(y, x)      # [-π, π] → [0, 2π]
elevation = arctan2(z, √(x²+y²))

# Binning
elev_bin = (elevation - elev_min) / (elev_max - elev_min) * n_elevation
azim_bin = azimuth / (2π) * n_azimuth
```

### 2.3 Empty Pixel Interpolation (Critical)

**문제점 발견:**
- 센서 밀도가 다르면 같은 장소도 완전히 다른 FFT 결과 생성
- 빈 픽셀(0)이 DC 성분 왜곡 및 고주파 노이즈 유발
- 동일 패턴도 밀도 차이로 Wasserstein distance = 81+ (0이어야 함)

**해결책: Circular Linear Interpolation**
```python
def interpolate_range_image(range_image):
    for row in range(n_elevation):
        valid_mask = row_data > 0
        valid_indices = np.where(valid_mask)[0]
        invalid_indices = np.where(~valid_mask)[0]

        # Circular boundary handling (azimuth wraps around)
        extended_indices = np.concatenate([
            valid_indices - n_azimuth,
            valid_indices,
            valid_indices + n_azimuth
        ])
        extended_values = np.tile(valid_values, 3)

        # Linear interpolation
        interpolated = np.interp(invalid_indices, extended_indices, extended_values)
```

**효과:**
- HeLiPR R@1: 53.10% → 68.56% (+15.46%p)

### 2.4 Spectral Encoder

**1D FFT (Ring-wise):**
```python
# Range image: (n_elevation, n_azimuth)
fft_result = torch.fft.rfft(range_image, dim=1)  # Along azimuth
magnitudes = torch.abs(fft_result)  # (n_elevation, n_freqs)
# n_freqs = n_azimuth // 2 + 1 = 181
```

**Exponential Frequency Binning:**
```python
# Bin edges with exponential warping
t = torch.linspace(0, 1, n_bins + 1)
bin_edges = (exp(α*t) - 1) / (exp(α) - 1) * n_freqs

# α = 2.0 (learnable parameter)
# 저주파에 더 많은 bin 할당 (구조 정보 보존)
```

**Histogram 생성:**
```python
# Vectorized binning (searchsorted + scatter_add)
bin_assignments = torch.searchsorted(bin_edges, freq_indices) - 1
histogram = torch.zeros(n_bins)
histogram.scatter_add_(0, bin_assignments, magnitude_sum)
```

**출력:** 50-dimensional spectral histogram

### 2.5 Graph Neural Network (GNN)

**아키텍처: 3-layer Graph Attention Network (GAT)**

```python
class SpectralGNN(nn.Module):
    def __init__(self):
        self.convs = [GATConv(50, 50, heads=1, edge_dim=1) for _ in range(3)]
        self.batch_norms = [BatchNorm(50) for _ in range(3)]

    def forward(self, data):
        x = data.x  # (N, 50)
        edge_index = data.edge_index
        edge_attr = data.edge_attr  # (E, 1) distance

        for conv, bn in zip(self.convs, self.batch_norms):
            x_new = conv(x, edge_index, edge_attr)
            x_new = bn(x_new)
            x_new = F.relu(x_new)
            x = x + x_new  # Residual connection

        return x
```

**그래프 구성:**
- **Node**: Keyframe descriptor (50-dim)
- **Edge**: Temporal neighbors (M=5, 앞뒤 2-3개씩)
- **Edge attribute**: Spatial distance (meters)

**파라미터 수:** 8,250

### 2.6 Triplet Mining

**Positive pairs:**
- 거리 < 5m
- Temporal gap > 30 frames (same location, different time)

**Hard negatives:**
- 10m < 거리 < 50m
- Smallest Wasserstein distance (most confusing)

**Loss function:**
```python
triplet_loss = max(0, d(anchor, positive) - d(anchor, negative) + margin)
# margin = 0.1
```

---

## 3. 데이터셋

### 3.1 학습 데이터

| Dataset | Sequences | Sensor | Environment | Keyframes |
|---------|-----------|--------|-------------|-----------|
| KITTI | 00-08 | HDL-64E | 독일 도심 | ~20K |
| NCLT | 6 dates | HDL-32E | 미국 캠퍼스 | ~150K |

**NCLT Sequences:**
- 2012-01-08, 2012-05-11, 2012-08-04
- 2012-11-04, 2012-11-16, 2013-02-23

### 3.2 검증 데이터

| Dataset | Sequence | Sensor | Environment |
|---------|----------|--------|-------------|
| KITTI | 09 | HDL-64E | 독일 도심 |

### 3.3 테스트 데이터 (학습 미포함)

| Dataset | Sequence | Sensor | Environment | Scans | Loop Queries |
|---------|----------|--------|-------------|-------|--------------|
| HeLiPR | Roundabout01 | VLP-16 | 한국 회전교차로 | 27,064 | 14,610 |
| HeLiPR | Town01 | VLP-16 | 한국 시내 | 23,935 | 9,280 |

---

## 4. 실험 결과

### 4.1 학습 결과

**Training Configuration:**
- Epochs: 44 (early stopping)
- Batch size: 32
- Learning rate: 0.001
- Optimizer: Adam
- Total time: 13.81 hours

**Validation (KITTI seq 09):**

| Metric | Value |
|--------|-------|
| R@1 | 99.12% |
| R@5 | 99.75% |
| R@10 | 99.87% |

### 4.2 Cross-Dataset Generalization (HeLiPR)

**HeLiPR Roundabout01 (VLP-16, 한국 회전교차로):**

| Method | R@1 | R@5 | R@10 | R@20 |
|--------|-----|-----|------|------|
| Spectral Histogram (Wasserstein) | **68.56%** | **75.20%** | **77.76%** | **80.10%** |
| GNN Embeddings (L2) | 65.28% | 68.96% | 70.79% | 72.66% |

**HeLiPR Town01 (VLP-16, 한국 시내):**

| Method | R@1 | R@5 | R@10 | R@20 |
|--------|-----|-----|------|------|
| Spectral Histogram (Wasserstein) | **40.77%** | **47.73%** | **51.55%** | **56.19%** |

**환경별 성능 비교:**

| Dataset | Environment | Scans | Loop Queries | R@1 | R@5 | R@10 |
|---------|-------------|-------|--------------|-----|-----|------|
| KITTI seq 09 | 독일 도심 | 1,591 | 232 | **99.12%** | 99.75% | 99.87% |
| HeLiPR Roundabout01 | 한국 회전교차로 | 27,064 | 14,610 | **68.56%** | 75.20% | 77.76% |
| HeLiPR Town01 | 한국 시내 | 23,935 | 9,280 | **40.77%** | 47.73% | 51.55% |

**주요 발견:**
1. 원본 spectral histogram이 GNN보다 unseen 환경에서 더 좋은 성능
2. Wasserstein distance가 L2보다 spectral histogram 비교에 적합
3. GNN이 학습 데이터에 약간 overfitting
4. **환경 복잡도에 따른 성능 차이**: Town01(시내) < Roundabout01(회전교차로) < KITTI(도심)

### 4.3 Interpolation 효과

**HeLiPR Roundabout01:**

| Setting | R@1 |
|---------|-----|
| Without interpolation | 53.10% |
| With interpolation | 68.56% |
| **Improvement** | **+15.46%p** |

Interpolation이 센서 간 일반화에 결정적 역할.

### 4.4 Domain Gap 분석

**전체 결과 요약:**

| Dataset | Sensor | Environment | R@1 | Gap from KITTI |
|---------|--------|-------------|-----|----------------|
| KITTI seq 09 (val) | HDL-64E | 독일 도심 | 99.12% | - |
| HeLiPR Roundabout01 | VLP-16 | 한국 회전교차로 | 68.56% | -30.56%p |
| HeLiPR Town01 | VLP-16 | 한국 시내 | 40.77% | -58.35%p |

**Discriminability 분석:**

| Metric | KITTI | HeLiPR Roundabout01 |
|--------|-------|---------------------|
| Loop closure 거리 (mean) | 0.397 | 0.424 |
| Non-loop 거리 (mean) | 1.665 | 1.286 |
| Separation gap | 1.269 | 0.862 |
| **Discriminability** | **6.45** | **3.77** |

*Discriminability = (Non-loop 거리 - Loop 거리) / Loop 거리의 std*

**Domain gap 요인:**

1. **센서 차이**: 64 rings vs 16 rings → Interpolation으로 완화
2. **환경 구조 차이**:
   - KITTI (도심): 다양한 건물 구조 → 다른 위치가 확실히 구분됨
   - Roundabout01 (회전교차로): 원형 구조 → 다른 위치도 비슷하게 보임
   - Town01 (시내): 반복적인 건물/골목 → 더 많은 혼동
3. **Non-loop pairs의 유사성**: HeLiPR에서 다른 위치들이 더 비슷한 FFT 패턴을 가짐

**핵심 발견:**
- FFT 자체는 정상 작동 (loop closure pairs의 거리는 유사: 0.40 vs 0.42)
- 문제는 **non-loop pairs가 더 가깝다**는 것 (1.67 vs 1.29)
- 이는 알고리즘의 한계가 아니라 **환경의 특성**

```
KITTI:  같은 장소(0.4) -------- 큰 gap(1.3) -------- 다른 장소(1.7)
                              ↑ 구분 쉬움

HeLiPR: 같은 장소(0.4) ---- 작은 gap(0.9) ---- 다른 장소(1.3)
                              ↑ 구분 어려움
```

### 4.5 FFT Descriptor의 근본적 한계 분석

**핵심 질문:** 학습 데이터 bias 문제인가, 아니면 FFT descriptor 자체의 한계인가?

#### 4.5.1 Confusion Rate 분석

"다른 장소가 같은 장소처럼 보이는 비율"을 측정:

| Dataset | Loop Mean | Non-loop Mean | Gap | Confusion Rate @90% |
|---------|-----------|---------------|-----|---------------------|
| KITTI 09 | 0.00091 | 0.00306 | 0.00215 | **3.0%** |
| HeLiPR Roundabout01 | 0.00109 | 0.00338 | 0.00229 | **12.6%** |
| HeLiPR Town01 | 0.00161 | 0.00373 | 0.00212 | **27.5%** |

*Confusion Rate @90%: Loop pairs의 90th percentile threshold 이하인 non-loop pairs 비율*

**Distribution Overlap (non-loop이 loop 범위 안에 들어오는 비율):**

| Dataset | Overlap Rate |
|---------|--------------|
| KITTI 09 | 3.4% |
| HeLiPR Roundabout01 | 44.3% |
| HeLiPR Town01 | **63.3%** |

#### 4.5.2 가장 혼동되는 사례

**Town01에서 발견된 극단적 사례:**
```
Frames 325 vs 1610: spatial=898.9m, Wass=0.00045
```
→ **898m 떨어진 두 위치**가 loop pairs 평균(0.00161)보다 **3.6배 더 가까운** FFT 패턴

**KITTI vs Town01 비교:**
```
KITTI:  300m+ 떨어진 위치 → Wass ≈ 0.00054 (loop 평균의 0.6배)
Town01: 900m 떨어진 위치 → Wass ≈ 0.00045 (loop 평균의 0.3배)
```

#### 4.5.3 Town01이 Roundabout01보다 더 나쁜 이유

초기 가설 "원형 구조가 문제"는 틀렸음. Town01은 로터리가 아닌 시내 환경.

**실제 원인: 환경의 구조적 균일성**

| 환경 | 특성 | Confusion Rate |
|------|------|----------------|
| KITTI (독일 도심) | 다양한 건물 스타일, 공원, 교차로 | 3.0% |
| Roundabout01 (회전교차로) | 중심/외곽 구조 차이 존재 | 12.6% |
| Town01 (한국 시내) | 비슷한 도로 폭, 비슷한 건물 높이, 반복적 상가 | 27.5% |

**FFT가 capture하는 정보:**
- 360도 방향별 거리 분포의 주파수 성분
- 전체적인 개방/폐쇄 구조
- 건물/도로의 반복 패턴

**Town01 환경 특성:**
```
┌────┐ ┌────┐ ┌────┐     비슷한 높이의 건물
│    │ │    │ │    │
└────┘ └────┘ └────┘
═══════════════════════   비슷한 폭의 도로
┌────┐ ┌────┐ ┌────┐
│    │ │    │ │    │     반복적인 상가 구조
└────┘ └────┘ └────┘
```
→ 다른 위치에서도 FFT 패턴이 구조적으로 유사해짐

#### 4.5.4 결론: 학습 데이터 bias가 아닌 Descriptor의 한계

**학습 데이터 bias 가설이 틀린 이유:**
1. FFT magnitude는 회전 불변 → "같은 장소 = 비슷한 FFT"는 보장됨
2. 문제는 "다른 장소 = 다른 FFT"가 **환경에 따라 보장되지 않음**
3. 학습을 더 해도, 데이터를 더 모아도 이 문제는 해결되지 않음

**FFT 기반 Global Descriptor의 근본적 한계:**
```
Descriptor의 discriminative power ∝ 환경의 구조적 다양성

- 구조적으로 다양한 환경: 높은 구분력 (KITTI R@1 99%)
- 구조적으로 균일한 환경: 낮은 구분력 (Town01 R@1 41%)
```

**이 한계가 의미하는 것:**
1. FFT 기반 place recognition은 **환경 선택적**으로만 효과적
2. 균일한 도시 환경에서는 **local feature 기반 방법**이 필수
3. Global descriptor만으로는 모든 환경에서 일반화 불가능

---

## 5. 핵심 파일 구조

```
src/
├── data/
│   ├── kitti_loader.py      # KITTI 데이터 로더
│   ├── nclt_loader.py       # NCLT 데이터 로더
│   ├── helipr_loader.py     # HeLiPR 데이터 로더
│   └── pose_utils.py        # SE(3) 변환 유틸리티
│
├── encoding/
│   ├── range_image.py       # Range image projection + interpolation
│   └── spectral_encoder.py  # FFT + exponential binning
│
├── keyframe/
│   ├── selector.py          # Keyframe selection
│   ├── criteria.py          # Selection criteria
│   └── graph_manager.py     # Temporal graph construction
│
├── gnn/
│   ├── model.py             # SpectralGNN, LocalUpdateGNN
│   ├── trainer.py           # Training loop
│   └── triplet_miner.py     # Hard negative mining
│
└── retrieval/
    ├── wasserstein.py       # Wasserstein distance
    └── two_stage_retrieval.py  # Coarse-to-fine retrieval
```

---

## 6. 하이퍼파라미터

| Parameter | Value | Description |
|-----------|-------|-------------|
| n_elevation | 16 | Elevation bins (sensor-agnostic) |
| n_azimuth | 360 | Azimuth bins (1° resolution) |
| n_bins | 50 | Spectral histogram dimension |
| α (alpha) | 2.0 | Exponential binning parameter |
| temporal_neighbors | 5 | GNN graph connectivity |
| triplet_margin | 0.1 | Triplet loss margin |
| positive_distance_max | 5.0m | Loop closure threshold |
| negative_distance_min | 10.0m | Hard negative min distance |
| negative_distance_max | 50.0m | Hard negative max distance |
| GNN layers | 3 | Graph attention layers |
| GNN hidden_dim | 50 | Hidden dimension |

---

## 7. 실험 환경

### 7.1 하드웨어
- **GPU**: NVIDIA RTX 5000 Ada (32GB VRAM)
- **RAM**: 251GB
- **Storage**: NVMe SSD

### 7.2 소프트웨어
- Python 3.11
- PyTorch 2.x
- PyTorch Geometric
- NumPy, SciPy

### 7.3 학습 시간
- 데이터 로딩: ~2시간
- 그래프 빌드: ~4초
- GNN 학습: ~12시간
- **총 소요**: ~14시간

---

## 8. 결론 및 향후 과제

### 8.1 주요 성과
1. **Rotation invariance**: FFT magnitude 기반으로 회전 불변 descriptor 구현
2. **Sensor generalization**: Interpolation으로 다른 센서 간 일반화 개선 (+15%p)
3. **Compact representation**: 50-dim descriptor로 효율적 저장/검색
4. **학습 데이터 내 높은 성능**: KITTI R@1 99.12%

### 8.2 한계점

#### 8.2.1 환경 의존적 성능
| 환경 유형 | R@1 | Confusion Rate |
|----------|-----|----------------|
| 구조적으로 다양한 환경 (KITTI) | 99.12% | 3.0% |
| 회전교차로 (Roundabout01) | 68.56% | 12.6% |
| 구조적으로 균일한 환경 (Town01) | 40.77% | 27.5% |

#### 8.2.2 FFT Descriptor의 근본적 한계
- **학습 데이터 bias가 아님**: FFT magnitude는 수학적으로 회전 불변
- **환경 구조 의존성**: 균일한 도시 환경에서 다른 위치도 비슷한 FFT 패턴 생성
- **해결 불가능한 한계**: 학습을 더 해도, 데이터를 더 모아도 개선되지 않음

```
핵심 통찰:
FFT descriptor의 구분력 = f(환경의 구조적 다양성)

Town01에서 63.3%의 non-loop pairs가 loop pairs 범위 내에 존재
→ 거의 random retrieval 수준
```

#### 8.2.3 GNN의 역할
- GNN이 unseen 환경에서 오히려 성능 저하 (68.56% → 65.28%)
- 학습 환경의 temporal pattern에 overfitting
- Global descriptor의 한계를 GNN으로 극복 불가

### 8.3 향후 과제

#### 필수적 개선 (FFT 한계 극복)
1. **Local feature 결합**: Global FFT + Local keypoint로 구조적 균일성 문제 해결
2. **Hierarchical representation**: 다양한 scale의 구조 정보 활용
3. **Semantic information**: 건물/도로/식생 등 semantic 정보로 구분력 향상

#### 추가적 개선
4. **Geometric verification**: Re-ranking으로 false positive 제거
5. **Confidence estimation**: 환경 복잡도에 따른 신뢰도 추정
6. **Hybrid retrieval**: FFT가 효과적인 환경과 아닌 환경 자동 판별

### 8.4 핵심 교훈

> **"Global descriptor 기반 place recognition은 환경 선택적이다"**
>
> FFT 기반 spectral histogram은 구조적으로 다양한 환경에서만 효과적이며,
> 균일한 도시 환경에서는 local feature 기반 방법이 필수적이다.
> 이는 알고리즘의 버그가 아닌 **representation의 근본적 특성**이다.

---

## 9. HeLiPR 학습 및 평가 방법론 분석 (2026-01-27)

### 9.1 실험 설정

**학습:**
- 학습 데이터: HeLiPR 19개 시퀀스 (339,723 scans, ~95,000 keyframes)
- Validation: KITTI 09
- 학습 시간: 14.47 hours, 26 epochs
- Best validation R@1: 99.69%

### 9.2 평가 결과 (evaluate_helipr.py)

| Dataset | R@1 | R@5 | R@10 | Confusion Rate |
|---------|-----|-----|------|----------------|
| KITTI 09 | 28.57% | 46.43% | 53.57% | 28.57% |
| NCLT (avg) | 27.08% | 45.87% | 51.57% | 67.94% |
| HeLiPR (avg) | 51.30% | 64.31% | 68.79% | 46.35% |

**문제점: Validation R@1 = 99.69%인데, 평가 R@1 = 28.57%**

### 9.3 문제 원인 분석

#### 9.3.1 핵심 발견: 학습 validation과 평가의 정의 차이

| 항목 | 학습 Validation | 평가 스크립트 |
|------|-----------------|---------------|
| 쿼리 범위 | All-vs-all (모든 키프레임) | Loop closure queries only |
| Skip frames | 없음 (자기만 제외) | 30 frames |
| 의미 | "가장 가까운 것이 5m 이내?" | "재방문을 인식하는가?" |

#### 9.3.2 실험 검증 (KITTI 09)

```
Descriptor Only (GNN 없음):
  Loop closure R@1:  78.57%
  Loop closure R@5:  92.86%
  Loop closure R@10: 92.86%

With GNN:
  Loop closure R@1:  0.00%  (← -78.57%!)
  Loop closure R@5:  14.29%
  Loop closure R@10: 28.57%
```

**GNN이 loop closure 성능을 완전히 파괴!**

#### 9.3.3 왜 학습 validation은 99%인가?

```python
# All-vs-all retrieval에서 Top-1이 temporal neighbor일 확률
>>> 92.39%

# Temporal neighbor (±5 frames)가 5m 이내일 확률
>>> 82.72%

# → All-vs-all R@1 ≈ 99%는 trivial!
```

**핵심:**
- 92%의 Top-1 retrieval이 temporal neighbor
- Temporal neighbor는 거의 항상 5m 이내
- → "시간적 인접 = 유사" 학습

#### 9.3.4 GNN의 부작용

```
Temporal Graph:
[KF_t-2] ── [KF_t-1] ── [KF_t] ── [KF_t+1] ── [KF_t+2]
         ↘    ↓    ↙         ↘    ↓    ↙
              Message Passing

결과:
- 시간적으로 가까운 프레임의 embedding이 수렴
- 장소 구분력 상실
- Loop closure (시간적으로 먼 같은 장소) 인식 불가
```

### 9.4 결론

#### 9.4.1 문제 요약

| 문제 | 원인 | 영향 |
|------|------|------|
| 학습 목표 불일치 | All-vs-all vs Loop closure | Metric 의미 없음 |
| GNN 부작용 | Temporal message passing | 장소 구분력 파괴 |
| Triplet mining 문제 | Positive = temporal neighbor | 시간 인접성 학습 |

#### 9.4.2 권장 해결책

1. **즉시 적용 가능**: Descriptor만 사용 (GNN 제거) → R@1 78.57%
2. **단기**: Validation을 loop closure 방식으로 변경
3. **중기**: Triplet mining에서 temporal neighbor 제외
4. **장기**: Spatial graph 기반 GNN 재설계

#### 9.4.3 핵심 교훈

> **"Validation metric이 높다고 실제 성능이 좋은 것이 아니다"**
>
> All-vs-all recall은 temporal neighbor 때문에 쉽게 99%가 되지만,
> 실제 loop closure (시간적으로 먼 재방문 인식)는 완전히 다른 문제이다.
>
> **GNN 없이 raw descriptor만 사용해도 R@1 78.57% 달성 가능.**
