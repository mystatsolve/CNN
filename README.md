# CNN 이미지 분류 모델 - 가상 도형 이미지

PyTorch CNN을 활용하여 도형을 분류하는 프로젝트입니다.
실제 데이터셋 대신 **PIL로 가상 이미지를 직접 생성**하여 사용합니다.

### 기본 제공 클래스 (5개, 자유롭게 추가/삭제 가능)

| 클래스 | 영문 |
|--------|------|
| 원 | circle |
| 사각형 | square |
| 삼각형 | triangle |
| 마름모 | diamond |
| 십자 | cross |

> `CLASS_NAMES` 리스트만 수정하면 클래스 수가 자동 확장됩니다.

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
    │   ├── triangle/
    │   ├── diamond/
    │   └── cross/
    └── val/
        ├── circle/
        ├── square/
        ├── triangle/
        ├── diamond/
        └── cross/
```

## 파이프라인

| 단계 | 내용 | 설명 |
|------|------|------|
| 1 | 데이터 생성 | PIL로 랜덤 위치·크기·색상의 도형 이미지 생성 |
| 2 | Dataset / DataLoader | 이미지를 텐서로 변환하고 미니배치 단위로 로딩 |
| 3 | CNN 모델 정의 | Conv→BN→ReLU→Pool ×3 → AdaptiveAvgPool → FC |
| 4 | 학습 | 15 에폭, Adam, lr=0.001, CrossEntropyLoss |
| 5 | 시각화 | 학습 곡선 (Loss / Accuracy) 그래프 |
| 6 | 추론 | 다양한 크기의 새 이미지 → 리사이즈 없이 바로 추론 |

## 모델 아키텍처

```
입력: (3, H, W) ← 임의 크기 가능

  [Block 1] Conv2d(3→16) → BatchNorm → ReLU → MaxPool
  [Block 2] Conv2d(16→32) → BatchNorm → ReLU → MaxPool
  [Block 3] Conv2d(32→64) → BatchNorm → ReLU → MaxPool

  AdaptiveAvgPool2d(4, 4)  ← 어떤 크기든 (64, 4, 4)로 통일

  Flatten → Linear(1024→128) → ReLU → Dropout(0.3) → Linear(128→N)

출력: N개 클래스 로짓 (N = CLASS_NAMES 길이)
```

### 핵심 설계

- **AdaptiveAvgPool2d**: 입력 이미지 크기에 상관없이 동작 (32×32, 128×128, 256×256 등)
- **num_classes 자동 설정**: `CLASS_NAMES` 길이에 맞춰 모델 출력층 자동 조절

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

### 클래스 추가 방법

1. `CLASS_NAMES`에 새 클래스명 추가
2. `CLASS_KR`에 한글 매핑 추가
3. `generate_shape_image()`에 `elif` 블록으로 그리기 로직 추가
4. 나머지는 자동 확장됨

## 기술 스택

- **PyTorch** - 딥러닝 프레임워크
- **PIL (Pillow)** - 가상 이미지 생성
- **NumPy** - 배열 연산
- **Matplotlib** - 시각화
