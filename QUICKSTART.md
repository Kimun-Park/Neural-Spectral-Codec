# Quick Start Guide

빠른 프로토타입을 위한 최소 설정 가이드입니다.

## 1. 최소 데이터 다운로드 (Option A: 단일 시퀀스)

### Option A-1: KITTI 수동 다운로드
```bash
# 1. KITTI 웹사이트에서 다운로드
# https://www.cvlibs.net/datasets/kitti/eval_odometry.php

# 2. 최소 다운로드 (sequence 00만, ~4.5GB)
# - data_odometry_velodyne.zip에서 sequences/00/velodyne/ 폴더
# - data_odometry_poses.zip에서 sequences/00/poses.txt

# 3. 압축 해제
mkdir -p data/kitti/sequences/00
# velodyne 폴더와 poses.txt를 data/kitti/sequences/00/에 복사
```

### Option A-2: wget으로 자동 다운로드
```bash
# 전체 다운로드 (80GB) - 시간이 오래 걸림
mkdir -p data/kitti
cd data/kitti

# Velodyne scans
wget https://s3.eu-central-1.amazonaws.com/avg-kitti/data_odometry_velodyne.zip

# Poses (ground truth)
wget https://s3.eu-central-1.amazonaws.com/avg-kitti/data_odometry_poses.zip

# 압축 해제
unzip data_odometry_velodyne.zip
unzip data_odometry_poses.zip

cd ../..
```

## 2. 더미 데이터로 테스트 (Option B: 데이터 없이 테스트)

데이터 다운로드 없이 바로 테스트하려면:

```bash
# 더미 데이터 생성 스크립트 실행
python scripts/create_dummy_data.py --output data/kitti --num_frames 100
```

## 3. 빠른 프로토타입 실행

```bash
# 의존성 설치
pip install -r requirements.txt

# 프로토타입 테스트 (500 프레임만)
python quick_prototype.py --sequence 00 --max_frames 500
```

예상 출력:
```
==============================================================
NEURAL SPECTRAL CODEC - QUICK PROTOTYPE
==============================================================
Data directory: data/kitti
Sequence: 00
Max frames: 500

Loading KITTI sequence 00...
Loaded 4541 frames

==============================================================
TEST 1: Spectral Encoding
==============================================================
Point cloud shape: (124668, 4)
Descriptor shape: torch.Size([50])
Encoding time: 8.45ms
Target: <10ms ✓

Testing rotation invariance...
Max difference after rotation: 0.000234
Rotation invariance: ✓

==============================================================
TEST 2: Keyframe Selection
==============================================================
Processing 500 frames...
  Processed 100/500 frames...
  Processed 200/500 frames...
  Processed 300/500 frames...
  Processed 400/500 frames...
  Processed 500/500 frames...

Results:
  Total scans: 500
  Keyframes: 48
  Compression: 10.4x
  Keyframe rate: 0.96 Hz
  Target rate: ~1Hz ✓

==============================================================
TEST 3: GNN Forward Pass
==============================================================
Building graph with 48 keyframes...
Graph: 48 nodes, 230 edges
GNN parameters: 15,300
Forward pass time: 12.34ms
Output shape: torch.Size([48, 50])

==============================================================
TEST 4: Two-Stage Retrieval
==============================================================
Adding 38 keyframes to database...
Querying database...

Results:
  Query time: 3.21ms
  Candidates found: 5

Top 3 candidates:
    1. Database idx 12, Wasserstein distance: 0.0234
    2. Database idx 25, Wasserstein distance: 0.0456
    3. Database idx 8, Wasserstein distance: 0.0789

Testing geometric verification...
  Total time (with GICP): 145.67ms
  Verified candidates: 2
    1. Fitness: 0.456, RMSE: 0.234m
    2. Fitness: 0.378, RMSE: 0.412m

==============================================================
PROTOTYPE TEST COMPLETE
==============================================================

Next steps:
  1. Download full KITTI sequences 00-09 for training
  2. Run: python src/pipeline.py --config configs/training.yaml --mode train
  3. Evaluate on validation sequence

All core components working! ✓
```

## 4. 시스템 요구사항

### 최소 사양 (프로토타입)
- CPU: 4 cores
- RAM: 8GB
- GPU: Optional (CPU로도 가능)
- 디스크: 10GB (sequence 00 하나만)

### 권장 사양 (전체 학습)
- CPU: 8+ cores
- RAM: 16GB
- GPU: NVIDIA GPU 4GB+ VRAM
- 디스크: 50GB

## 5. 다음 단계

### 프로토타입 성공 후:

```bash
# 1. 전체 시퀀스 다운로드 (sequences 00-09, ~37GB)
# 위의 wget 명령어 사용

# 2. 전체 학습 실행
python src/pipeline.py --config configs/training.yaml --mode train

# 3. 추론 테스트
python src/pipeline.py --config configs/inference.yaml --mode inference
```

## 6. 문제 해결

### KITTI 데이터가 없는 경우:
```bash
# 더미 데이터로 테스트
python scripts/create_dummy_data.py
python quick_prototype.py
```

### GPU 메모리 부족:
```yaml
# configs/default.yaml 수정
system:
  device: "cpu"  # GPU 대신 CPU 사용
```

### PyTorch Geometric 설치 오류:
```bash
# CPU 버전으로 설치
pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cpu
pip install torch-geometric==2.4.0
```

## 7. 빠른 성능 확인

```bash
# 인코딩 속도만 테스트
python -c "
from quick_prototype import test_encoding
from data.kitti_loader import KITTILoader
loader = KITTILoader('data/kitti', '00')
test_encoding(loader, max_frames=10)
"
```

## 8. 예상 소요 시간

```
프로토타입 테스트 (500 frames):
  - 데이터 로드: 5초
  - 인코딩 테스트: 1초
  - 키프레임 선택: 30초
  - GNN 테스트: 2초
  - 검색 테스트: 5초
  총: ~1분

전체 학습 (sequences 00-08):
  - 데이터 전처리: 10분
  - 키프레임 선택: 30분
  - GNN 학습 (50 epochs): 2시간
  총: ~3시간
```

## 9. 데이터 크기 비교

| 옵션 | 시퀀스 | 크기 | 프레임 수 | 용도 |
|------|--------|------|-----------|------|
| Minimal | 00 | 4.5GB | 4,541 | 프로토타입 |
| Small | 00, 05 | 7GB | 7,302 | 빠른 테스트 |
| Training | 00-08 | 35GB | ~20,000 | 전체 학습 |
| Full | 00-10 | 45GB | ~25,000 | 검증 포함 |

시작은 **Minimal**로 하시는 것을 권장합니다!
