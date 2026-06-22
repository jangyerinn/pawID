#  PawID
**대조 학습 기반 유기견 비문(코 무늬) 생체 식별 시스템**

> Triplet Loss, IQA, Similarity-CAM을 활용한 유기견 신원 확인 시스템  
> 딥러닝실습 24012480 장예린

---

## 빠른 데모 (평가자용)

GitHub 저장소에는 학습된 모델 가중치와 데모용 등록 DB(50마리)가 이미 포함되어 있어, **재학습이나 데이터 준비 없이 `app.py`만 실행하면 바로 식별 기능을 체험할 수 있다.**

```bash
git clone https://github.com/jangyerinn/PawID.git
cd PawID
pip install -r requirements.txt
python app.py
```

> 경로는 `app.py` 위치를 기준으로 자동 인식되므로, 어떤 폴더에 클론하든 동일하게 작동한다.

아래 "설치 및 실행" 절차는 모델을 처음부터 직접 재현·재학습하고 싶을 때만 필요하다.

---

## 프로젝트 구조

```
PawID/
├── app.py                     # Gradio 웹앱 (최종 버전)
├── requirements.txt           # 의존성 패키지
├── 1_label.md                 # Roboflow 라벨링 가이드
├── 0_prepare_data.py          # CSV 기반 데이터 전처리
├── 2_finetune.py              # YOLOv8n 코 탐지기 파인튜닝
├── model.py                   # EfficientNet-B0 + Embedding Head
├── train.py                   # Triplet Loss + PK Sampler + Semi-Hard Negative Mining
├── 3_train.py                 # Baseline → Triplet Loss 학습
├── train_arcface.py           # ArcFace 비교실험
├── train_texture.py           # CLAHE+160px 텍스처 강화 학습 (최종 채택 모델)
├── build_db.py                # FAISS 벡터 DB (등록 / 검색)
├── eval_and_cam.py            # 성능 평가 + Similarity-CAM (XAI, 코사인 유사도 역전파)
├── val_verify.py              # Val 쌍 기반 TAR/FRR 검증
├── verify_texture.py          # 텍스처 모델 FAR/TAR 검증, ROC Curve, EER 분석
├── check_far.py               # FAR(오인식률) 측정
├── ensemble_test.py           # 단순 앙상블 비교 실험
├── ensemble_weighted_test.py  # 가중치 스캔 앙상블 실험
├── multi_photo_test.py        # 다중 사진 평균 효과 검증
├── test_identify_accuracy.py  # 실제 DB 기준 Rank-1/5 Accuracy 검증
├── bulk_register_val.py       # validation 쌍 일괄 등록 도구
├── check_db.py                # DB 등록 내용 확인 도구
├── models/                    # 학습된 가중치 저장 위치
└── db/                        # FAISS 인덱스 및 메타데이터
```



---

## 개발 일정 (8주)

| 주차 | 작업 | 파일 |
|------|------|------|
| 1주차 | 환경 세팅, CVPR 데이터 다운로드, 코 라벨링 200장 (Roboflow), IQA 모듈 구현 | `app.py` (IQA 로직) |
| 2주차 | YOLOv8n 코 탐지기 파인튜닝, 코 크롭 파이프라인 구축 | `2_finetune.py` |
| 3주차 | EfficientNet-B0 + Embedding Head 구현, Baseline 학습 | `model.py`, `3_train.py` |
| 4주차 | PK Sampler 구현, Triplet Loss + Semi-Hard Negative Mining 적용 | `train.py` |
| 5주차 | 데이터 증강, FAISS 인덱스 구축, IQA 파라미터 최적화 | `build_db.py` |
| 6주차 | Similarity-CAM 구현 (코사인 유사도 역전파 방식) | `app.py` (CAM 로직), `eval_and_cam.py` |
| 7주차 | 시연 DB 구축, Gradio 웹앱 완성 | `app.py` |
| 8주차 | 버그 수정, ROC Curve 분석, 최종 보고서 | `eval_and_cam.py`, `verify_texture.py`, `test_identify_accuracy.py` |

---

## 설치 및 실행

### 1. 환경 설치
```bash
conda create -n pawid python=3.10
conda activate pawid
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

### 2. 데이터 준비
- CVPR 2022 Pet Biometric Challenge Dataset: https://www.kaggle.com/datasets/zekunn/pet-biometric-challenge/data
- YOLOv8 라벨링: Roboflow에서 dog_nose 클래스 직접 라벨링 (200장, Unsplash·Stanford Dogs Dataset에서 수집)

데이터 전처리:
```bash
python 0_prepare_data.py
```

### 3. YOLOv8n 파인튜닝 (2주차)
```bash
python 2_finetune.py --data_yaml <data.yaml 경로> --epochs 50
```

### 4. 비문 인식 모델 학습 (3~4주차)
```bash
# Baseline / Triplet+PK+SemiHard 비교실험
python 3_train.py --mode semi_hard

# ArcFace 비교실험
python train_arcface.py

# 텍스처 강화 (최종 채택 모델)
python train_texture.py --epochs 150
```

### 5. FAISS DB 구축
```bash
python build_db.py --model models/semi_hard_texture.pt
```

### 6. 성능 검증 (8주차)
```bash
python verify_texture.py
python test_identify_accuracy.py
```

### 7. Gradio 앱 실행 (7주차)
```bash
python app.py
```

---

## 핵심 기술 및 선택 근거

### IQA 모듈 (Laplacian Variance)
- Petnow 등 상용 앱은 딥러닝 AI를 내장해 품질 검증 → 고비용·폐쇄적
- PawID: 수학 알고리즘(Laplacian Variance)으로 초경량 IQA 독자 설계
- 별도 학습 없이 흔들림·조도를 수학적으로 계산해 불량 데이터 원천 차단

### Contrastive Learning
- 비문 식별은 분류(Classification) 문제가 아님
- 새 강아지 등록 시 재학습 불필요 → 벡터 DB에 추가하면 끝
- 수천 마리로 확장 가능

### PK Sampler + Semi-Hard Negative Mining
- 일반 DataLoader: 배치 내 Positive 쌍이 우연히 들어갈 확률 낮아 Loss 계산 부실
- PK Sampler: P마리 × K장으로 Positive 쌍 항상 보장 (P=8, K=4, 배치=32)
- Semi-Hard: d(A,P) < d(A,N) < d(A,P)+margin → 가장 안정적이고 효과적
- Ref: Hermans et al. (2017), arXiv:1703.07737

### CLAHE 텍스처 강화 (최종 채택)
- Similarity-CAM으로 모델이 코 형태에 의존하는 경향을 확인
- CLAHE 전처리 + 입력 해상도 축소(224→160px)로 재학습
- FAR 12.25%→7.15%로 거의 절반 감소

### Similarity-CAM
- 임베딩 모델에 일반 Grad-CAM 적용 시 에러 (클래스 출력층 없음)
- 코사인 유사도 점수를 역전파 타겟으로 사용하는 방식으로 변형
- "코의 이 부분 무늬 패턴이 일치해서 같은 강아지로 판단" 시각화

### 다중 사진 등록 평균 (최종 채택)
- 강아지 1마리당 여러 장 등록 시 임베딩 평균
- Rank-1 86.0%→90.5%, Rank-5 95.5%→98.5% 개선 확인

---

## GitHub 저장소

https://github.com/jangyerinn/PawID

전체 소스코드, 학습된 모델 가중치 7종, 데모용 등록 DB(50마리)를 공개 배포함.

---

## 참고문헌

1. Hermans, A., Beyer, L., & Bastian, B. (2017). In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.
2. Tan, M., & Le, Q. V. (2019). EfficientNet: Rethinking Model Scaling for CNNs. ICML.
3. Schroff, F., et al. (2015). FaceNet: A Unified Embedding for Face Recognition and Clustering. CVPR.
4. Johnson, J., et al. (2019). Billion-scale similarity search with GPUs (FAISS). IEEE Transactions on Big Data.
5. Selvaraju, R. R., et al. (2017). Grad-CAM. ICCV.
6. Jocher, G., et al. (2023). YOLO by Ultralytics (YOLOv8).
7. Müller, S., et al. (2019). Dog Nose Biometrics: A Systematic Review of Pattern Uniqueness.
