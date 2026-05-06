# DARAM Coil — 산업용 코일 표면 결함 검출 시스템

PatchCore 기반 이상탐지(Anomaly Detection) 파이프라인을 구축하여 산업용 코일의 표면 결함을 자동 검출하는 시스템입니다.  
정상 샘플만으로 학습하며, **2-class(OK/NG)** 및 **3-class(OK/REVIEW/NG)** 분류를 모두 지원합니다.

---

## 프로젝트 목표

### 3-class 분류 구조
미검(NG → OK 오판)을 **0건** 유지하면서, OK·NG를 60~70% 자동 판정하고 나머지는 전문가 육안 검사(REVIEW)로 넘기는 구조입니다.

| 조건 | 결과 |
|------|------|
| `score ≤ T_OK` | ✅ OK |
| `T_OK < score < T_NG` | 🔍 REVIEW |
| `score ≥ T_NG` | ❌ NG |

> 현재 자동 판정 성능: **약 40%** (목표 60~70%, 개발 중)

---

## 기술 스택

| 분류 | 내용 |
|------|------|
| **모델** | PatchCore (`wide_resnet50_2`, layer2/layer3) |
| **학습 프레임워크** | anomalib 0.7.x, PyTorch Lightning |
| **추론 최적화** | OpenVINO (NPU 지원) |
| **전처리** | OpenCV Letterbox (비율 유지, 고정 입력 256×384) |
| **이상 점수** | Memory Bank 기반 최근접 거리 (k-NN) |
| **시각화** | Anomaly Map Heatmap Overlay |

---

## 전체 파이프라인

```
[정상 이미지 학습 데이터]
        │
        ▼
┌──────────────────────────────┐
│  전처리                       │
│  Letterbox → 256×384         │
│  타일링 256×256(stride 256)  │
│  ImageNet 정규화              │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│  PatchCore 학습               │
│  ResNet50 feature 추출        │
│  (layer2, layer3)             │
│  Coreset Sampling (ratio=0.05)│
│  → Memory Bank 구축           │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│  Threshold 결정               │
│  2-class: 정확도 최대 T       │
│  3-class: 미검 0 기준 T_OK,   │
│           T_NG 설정           │
└──────────────┬───────────────┘
               │
        ┌──────┴──────┐
        ▼             ▼
┌─────────────┐ ┌──────────────────┐
│ PyTorch 추론 │ │ OpenVINO NPU 추론 │
│ Jupyter /   │ │ fe_wrap.xml +    │
│ .py 스크립트 │ │ last.ckpt        │
└─────────────┘ └──────────────────┘
        │             │
        └──────┬──────┘
               ▼
      OK / REVIEW / NG
    + Anomaly Heatmap 저장
```

---

## 폴더 구조

```
DARAM_Coil/
├── CODE/
│   ├── Patchcore04.ipynb            # 학습 + 평가 관리
│   ├── PatchCore_3class.ipynb       # 3-class 단일 이미지 추론
│   └── PatchCore_2class.ipynb       # 2-class 단일 이미지 추론
│
├── Patchcore_tilling/               # 학습 스크립트
│   ├── PatchCore-Train-Tiling-NoResize.py   # ✅ 권장 (메모리 효율화)
│   ├── PatchCore-Train-Tiling-CVLetterbox.py
│   ├── PatchCore-Train-Tiling-PPDensity.py  # Density-aware Coreset
│   ├── PatchCore-Train-Tiling.py
│   └── tiling_heatmap_gt.py         # 시각화 유틸
│
├── ver_OpenVino_maps/               # OpenVINO NPU 추론
│   ├── ver_OpenVino_2class_NPU/
│   │   ├── infer_2class.py          # 단일 이미지
│   │   ├── batch_infer_2class.py    # 배치 처리
│   │   └── model/                   # fe_wrap.xml + last.ckpt
│   └── ver_OpenVino_3class_NPU/
│       ├── infer_3class.py
│       ├── batch_infer_3class.py
│       ├── threshold_policy_3class.csv
│       └── model/
│
├── model/                           # 학습된 모델 체크포인트
│   └── PatchCore_0.05/last.ckpt
│
├── DATASET/                         # ⛔ 회사 기밀 — git 제외
└── READ_ME/
    └── README.md
```

---

## 사용 방법

### 1. 모델 학습

```bash
# 권장: 메모리 효율화 버전
python Patchcore_tilling/PatchCore-Train-Tiling-NoResize.py fit validate test viz
```

### 2. 단일 이미지 추론

```bash
# 2-class (NPU)
python ver_OpenVino_maps/ver_OpenVino_2class_NPU/infer_2class.py \
  --input sample.png --threshold 37.0 --device NPU

# 3-class (NPU)
python ver_OpenVino_maps/ver_OpenVino_3class_NPU/infer_3class.py \
  --input sample.png --device NPU
```

### 3. 배치 처리

```bash
python ver_OpenVino_maps/ver_OpenVino_2class_NPU/batch_infer_2class.py \
  --input-dir ./images --device NPU

python ver_OpenVino_maps/ver_OpenVino_3class_NPU/batch_infer_3class.py \
  --input-dir ./images --device NPU
```

---

## 현재 성능 (Patchcore03 기준)

### 2-class
| 지표 | 값 |
|------|-----|
| Test Accuracy | 88.42% |
| AUROC | 0.834 |
| Best F1 | 0.640 |
| Threshold | 47.40 |

### 3-class (미검 최소화 정책)
| 지표 | 값 |
|------|-----|
| 미검 (NG→OK) | **0건** |
| 과검 (OK→NG) | **0건** |
| 자동 판정 비율 | 22.11% |
| T_OK | 32.88 |
| T_NG | 56.50 |

### 추론 속도 (OpenVINO NPU)
| 단계 | 시간 |
|------|------|
| 전처리 | ~0.02s |
| 추론 (NPU) | ~0.10s |
| 후처리 | <0.01s |
| **이미지당 합계** | **~0.13s** |

---

## 주요 설계 결정

| 항목 | 선택 | 이유 |
|------|------|------|
| Backbone | `wide_resnet50_2` | ImageNet pretrained, 충분한 receptive field |
| Feature Layer | layer2 + layer3 | 다양한 스케일의 feature 확보 |
| Coreset Ratio | 0.05 | 메모리 5%로 정확도 손실 최소화 |
| 전처리 | Letterbox | 강제 resize 왜곡 없이 비율 유지 |
| OpenVINO 적용 범위 | Feature Extractor만 | Anomaly Map 왜곡 방지 |
| Postprocessing | PyTorch CPU | 안정성 우선 |

---

## 진행 현황

- [x] PatchCore 학습 파이프라인 구축
- [x] Letterbox 전처리 적용
- [x] 타일링(Tiling) 기반 대용량 이미지 처리
- [x] 2-class 분류 (OK/NG)
- [x] 3-class 분류 (OK/REVIEW/NG) — 미검 0건 달성
- [x] OpenVINO NPU 변환 및 추론
- [ ] 3-class 자동 판정 성능 향상 (현재 40% → 목표 60~70%)
- [ ] 현장 적용 및 실시간 처리 최적화

---

## 환경 요구사항

```
Python 3.9 / 3.10
anomalib >= 0.7
pytorch-lightning >= 2.0
torch, torchvision
openvino >= 2024.0
opencv-python
albumentations
numpy, pandas, pillow
```
