# CNN 이미지 분류 모델 - 가상 도형 이미지

PyTorch CNN을 활용하여 **원(Circle), 사각형(Square), 삼각형(Triangle)** 3가지 도형을 분류하는 프로젝트입니다.
실제 데이터셋 대신 **PIL로 가상 이미지를 직접 생성**하여 사용합니다.

## 프로젝트 구조

```
CNN/
├── cnn_image_classifier.ipynb   # 전체 파이프라인 노트북 (실행 결과 포함)
├── .gitignore
├── README.md
└── shape_data/                  # 데이터 폴더 (노트북 실행 시 이미지 자동 생성)
    ├── train/
    │   ├── circle/
    │   ├── square/
    │   └── triangle/
    └── val/
        ├── circle/
        ├── square/
        └── triangle/
```

## 파이프라인

| 단계 | 내용 | 설명 |
|------|------|------|
| 1 | 데이터 생성 | PIL로 랜덤 위치·크기·색상의 도형 이미지 생성 (train 1,500장 + val 300장) |
| 2 | Dataset / DataLoader | 이미지를 텐서로 변환하고 미니배치 단위로 로딩 |
| 3 | CNN 모델 정의 | Conv→BN→ReLU→Pool ×3 → FC (약 55만 파라미터) |
| 4 | 학습 | 15 에폭, Adam, lr=0.001, CrossEntropyLoss |
| 5 | 시각화 | 학습 곡선 (Loss / Accuracy) 그래프 |
| 6 | 추론 | 새 이미지 생성 → 모델 예측 → 결과 시각화 |

## 모델 아키텍처

```
입력: (3, 64, 64)

  [Block 1] Conv2d(3→16) → BatchNorm → ReLU → MaxPool  → (16, 32, 32)
  [Block 2] Conv2d(16→32) → BatchNorm → ReLU → MaxPool  → (32, 16, 16)
  [Block 3] Conv2d(32→64) → BatchNorm → ReLU → MaxPool  → (64, 8, 8)

  Flatten → Linear(4096→128) → ReLU → Dropout(0.3) → Linear(128→3)

출력: 3개 클래스 로짓
```

## 실행 결과

- **Validation Accuracy: 100%** (Epoch 2부터 수렴)
- **추론 테스트: 9/9 정답 (100%)**

## 실행 방법

### 요구 사항

- Python 3.10+
- PyTorch, torchvision, numpy, Pillow, matplotlib

```bash
pip install torch torchvision numpy pillow matplotlib
```

### 실행

Jupyter Notebook에서 `cnn_image_classifier.ipynb`를 순서대로 실행하면 됩니다.
데이터 생성 → 학습 → 추론이 모두 노트북 내에서 자동으로 수행됩니다.

```bash
jupyter notebook cnn_image_classifier.ipynb
```

## 기술 스택

- **PyTorch** - 딥러닝 프레임워크
- **PIL (Pillow)** - 가상 이미지 생성
- **NumPy** - 배열 연산
- **Matplotlib** - 시각화
