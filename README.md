# CNN 이미지 분류 모델 - 실제 도형 이미지

PyTorch CNN을 활용하여 도형을 분류하는 프로젝트입니다.
`shape_data/` 폴더에 저장된 **실제 이미지 데이터**를 사용하여 학습 및 추론합니다.

### 분류 클래스 (폴더 구조에서 자동 감지)

| 클래스 | 영문 |
|--------|------|
| 원 | circle |
| 사각형 | square |
| 삼각형 | triangle |
| 마름모 | diamond |
| 십자 | cross |

> 클래스는 `shape_data/train/` 하위 폴더명에서 자동으로 감지되므로, 폴더를 추가/삭제하면 자동 확장됩니다.

## 프로젝트 구조

```
CNN/
├── cnn_image_classifier.ipynb   # 전체 파이프라인 노트북 (실행 결과 포함)
├── .gitignore
├── README.md
└── shape_data/                  # 실제 이미지 데이터
    ├── train/                   # 학습 데이터 (클래스당 500장)
    │   ├── circle/
    │   ├── square/
    │   ├── triangle/
    │   ├── diamond/
    │   └── cross/
    └── val/                     # 검증 데이터 (클래스당 100장)
        ├── circle/
        ├── square/
        ├── triangle/
        ├── diamond/
        └── cross/
```

## 파이프라인

| 단계 | 내용 | 설명 |
|------|------|------|
| 1 | 데이터 확인 | shape_data 폴더 구조 및 이미지 수 확인, 샘플 시각화 |
| 2 | Dataset / DataLoader | 실제 이미지를 텐서로 변환하고 미니배치 단위로 로딩 |
| 3 | CNN 모델 정의 | Conv→BN→ReLU→Pool ×3 → AdaptiveAvgPool → FC |
| 4 | 학습 | 15 에폭, Adam, lr=0.001, CrossEntropyLoss |
| 5 | 시각화 | 학습 곡선 (Loss / Accuracy) 그래프 |
| 6 | 추론 테스트 | 검증 이미지로 개별 추론 및 클래스별 확률 출력 |
| 7 | 전체 평가 | 전체 검증 세트 클래스별 정확도 평가 |

## 모델 아키텍처

```
입력: (3, H, W) ← 임의 크기 가능

  [Block 1] Conv2d(3→16) → BatchNorm → ReLU → MaxPool
  [Block 2] Conv2d(16→32) → BatchNorm → ReLU → MaxPool
  [Block 3] Conv2d(32→64) → BatchNorm → ReLU → MaxPool

  AdaptiveAvgPool2d(4, 4)  ← 어떤 크기든 (64, 4, 4)로 통일

  Flatten → Linear(1024→128) → ReLU → Dropout(0.3) → Linear(128→N)

출력: N개 클래스 로짓 (N = 자동 감지된 클래스 수)
```

### 핵심 설계

- **AdaptiveAvgPool2d**: 입력 이미지 크기에 상관없이 동작 (32×32, 128×128, 256×256 등)
- **자동 클래스 감지**: 폴더 구조에서 클래스명과 개수를 자동으로 읽어옴
- **다양한 이미지 포맷 지원**: `.png, .jpg, .jpeg, .bmp, .gif, .tiff, .webp`

## 실행 방법

### 요구 사항

- Python 3.10+
- PyTorch, numpy, Pillow, matplotlib

```bash
pip install torch torchvision numpy pillow matplotlib
```

### 데이터 준비

`shape_data/train/` 및 `shape_data/val/` 하위에 클래스별 폴더를 만들고 이미지를 넣으세요.

### 실행

Jupyter Notebook에서 `cnn_image_classifier.ipynb`를 순서대로 실행하면 됩니다.

```bash
jupyter notebook cnn_image_classifier.ipynb
```

### 클래스 추가 방법

1. `shape_data/train/새클래스명/` 폴더에 학습 이미지 추가
2. `shape_data/val/새클래스명/` 폴더에 검증 이미지 추가
3. 노트북을 처음부터 다시 실행 (클래스 자동 감지)

## 기술 스택

- **PyTorch** - 딥러닝 프레임워크
- **PIL (Pillow)** - 이미지 로딩 및 전처리
- **NumPy** - 배열 연산
- **Matplotlib** - 시각화
