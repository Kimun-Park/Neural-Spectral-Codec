# Keyframe Selection 상세 설계

> 이 문서는 [overall_approach.md](overall_approach.md)의 Stage 4 (Keyframe Selection)의 상세 내용을 다룹니다.

---

## 1. 개요

### 1.1 목적

10Hz LiDAR 스캔에서 **중복을 제거**하고 대표적인 keyframe만 선택합니다. 저장 공간과 계산 비용을 절약하면서 place recognition 성능을 유지합니다.

### 1.2 설계 원칙

| 원칙 | 설명 |
|-----|------|
| Redundancy Removal | 유사한 스캔 제거 |
| Coverage Guarantee | 중요한 지점은 반드시 포함 |
| Adaptive Selection | 환경 변화에 따라 적응적 선택 |

---

## 2. 4-Criterion Selection Strategy

다음 4가지 기준 중 **하나라도 만족**하면 keyframe으로 선택 (OR logic):

| 기준 | 조건 | 기본 임계값 | 목적 |
|-----|------|------------|------|
| **Distance** | $d > d_{th}$ | 0.5m | 충분히 이동했을 때 |
| **Rotation** | $\theta > \theta_{th}$ | 15° | 시점 변화가 클 때 |
| **Geometric Novelty** | IoU < $o_{th}$ | 0.7 | 새로운 환경일 때 |
| **Temporal** | $\Delta t > t_{th}$ | 5s | 오래 대기했을 때 |

### 2.1 Distance Criterion

$$d = \lVert T_{\text{current}}[:3, 3] - T_{\text{last}}[:3, 3] \rVert_2$$

### 2.2 Rotation Criterion

$$\theta = \arccos\left(\frac{\text{trace}(R_{\text{rel}}) - 1}{2}\right)$$

### 2.3 Geometric Novelty (IoU)

$$\text{IoU} = \frac{|V_{\text{current}} \cap V_{\text{last}}|}{|V_{\text{current}} \cup V_{\text{last}}|}$$

Voxel size: 0.2m

---

## 3. Selection Logic

### 3.1 OR Logic (기본)

Distance, Rotation, Temporal 중 하나라도 만족하면 선택

### 3.2 Early Termination

Geometric novelty 계산은 비용이 높으므로, 다른 기준이 만족되면 건너뜀

| 기준 | 복잡도 |
|-----|--------|
| Distance/Rotation/Temporal | $O(1)$ |
| Geometric | $O(N)$ |

---

## 4. 파라미터 튜닝

### 4.1 환경별 권장값

| 환경 | Distance | Rotation | Temporal |
|-----|----------|----------|----------|
| 도심 (저속) | 1.0m | 10° | 3s |
| 고속도로 | 3.0m | 15° | 5s |
| 주차장 (정밀) | 0.5m | 5° | 2s |

### 4.2 속도별 Distance Threshold

$$d_{\text{th}} = v \cdot \Delta t \cdot k$$

- $v$: 평균 속도 (m/s)
- $\Delta t$: 프레임 간격 (0.1s @ 10Hz)
- $k$: Reduction factor

**예시:** 30 km/h, 10% keyframe rate → $d_{th} \approx 8m$

---

## 5. 요약

| 구성요소 | 역할 |
|---------|------|
| **Distance Criterion** | 공간 이동 기반 선택 |
| **Rotation Criterion** | 시점 변화 기반 선택 |
| **Geometric Novelty** | 환경 변화 기반 선택 |
| **Temporal Criterion** | 시간 기반 최소 보장 |
| **OR Logic** | 하나라도 만족 시 선택 |

---

**문서 생성일:** 2026-01-28
