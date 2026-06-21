# PawID
**대조 학습 기반 유기견 비문(코 무늬) 생체 식별 시스템**

> Triplet Loss · IQA · Similarity-CAM을 활용한 유기견 신원 확인 시스템
> 딥러닝실습 24012480 장예린

---

## 프로젝트 구조

```
PawID/
├── README.md                 # 프로젝트 설명 및 실행 가이드
├── requirements.txt           # 의존성 패키지 목록
├── 1_label.md                 # Roboflow 라벨링 가이드
├── app.py                     # Gradio 웹앱 (최종 버전)
├── 0_prepare_data.py          # CSV 기반 데이터 전처리
├── 2_finetune.py              # YOLOv8n 코 탐지기 파인튜닝
├── model.py                   # 비문 인식 모델 정의 (3_train.py에서 사용)
├── train.py                   # PK Sampler·Triplet Loss 유틸 (3_train.py에서 사용)
├── 3_train.py                 # Triplet+PK+SemiHard 학습 (비교실험용)
├── train_arcface.py           # ArcFace 학습 (비교실험용)
├── train_texture.py           # CLAHE+160px 텍스처 강화 학습 (최종 채택 모델)
├── build_db.py                # FAISS DB 구축
├── eval_and_cam.py            # 성능 평가 + Similarity-CAM
├── val_verify.py              # Val 쌍 기반 TAR/FRR 검증
├── verify_texture.py          # 텍스처 모델 FAR/TAR 검증
├── check_far.py               # FAR(오인식률) 측정
├── ensemble_test.py           # 단순 앙상블 비교 실험
├── ensemble_weighted_test.py  # 가중치 스캔 앙상블 실험
├── multi_photo_test.py        # 다중 사진 평균 효과 검증
├── test_identify_accuracy.py  # 실제 DB 기준 Rank-1/5 정확도 검증
├── bulk_register_val.py       # validation 쌍 일괄 등록 도구
└── check_db.py                # DB 등록 내용 확인 도구
```

> 초기에 모듈형(`src/`) 구조로 만들었던 일부 파일은 이후 독립 실행 가능한 스크립트 구조로 다시 작성하면서 사용을 중단해 최종 제출에서는 제외했다.

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
- YOLOv8 라벨링: Roboflow에서 dog_nose 클래스 직접 라벨링 (약 200장, Unsplash·Stanford Dogs Dataset에서 수집)

데이터 전처리:
```bash
python 0_prepare_data.py
```

### 3. YOLOv8n 코 탐지기 파인튜닝
```bash
python 2_finetune.py --data_yaml <data.yaml 경로> --epochs 50
```

### 4. 비문 인식 모델 학습
```bash
# 비교실험용 (Triplet+PK+SemiHard, 원본 224px)
python 3_train.py --mode semi_hard

# ArcFace 비교실험
python train_arcface.py

# 텍스처 강화 버전 (최종 채택 모델)
python train_texture.py --epochs 150
```

### 5. FAISS DB 구축
```bash
python build_db.py --model models/semi_hard_texture.pt
```

### 6. 성능 검증
```bash
python verify_texture.py
python test_identify_accuracy.py
python check_far.py
python ensemble_weighted_test.py
python multi_photo_test.py
```

### 7. Gradio 앱 실행
```bash
python app.py
```

---

## 핵심 기술 및 선택 근거

### IQA 모듈 (Laplacian Variance)
Petnow 등 상용 앱은 딥러닝 AI를 내장해 품질 검증을 하지만 고비용·폐쇄적이다. PawID는 수학 알고리즘(Laplacian Variance)으로 초경량 IQA 모듈을 독자 설계해, 별도 학습 없이 흔들림·조도를 수학적으로 계산해 불량 데이터를 차단한다.

### Contrastive Learning
비문 식별은 분류 문제가 아니다. 새 강아지 등록 시 재학습이 불필요하며, 벡터 DB에 임베딩을 추가하는 것만으로 확장 가능하다.

### Triplet Loss + PK Sampler + Semi-Hard Negative Mining
일반 DataLoader는 배치 내 Positive 쌍을 보장하지 못해 Loss 계산이 부실해진다. PK Sampler(P=8, K=4)로 Positive 쌍을 항상 보장하고, Semi-Hard Negative Mining으로 가장 효과적인 구간의 샘플을 학습에 사용한다.

### CLAHE 텍스처 강화 (최종 채택)
모델이 비문의 세밀한 텍스처보다 코의 전체적인 형태에 의존하는 경향을 Similarity-CAM으로 확인하고, CLAHE 전처리 + 입력 해상도 축소(224→160px)로 재학습해 FAR을 12.25%→7.15%로 줄였다.

### Similarity-CAM
임베딩 모델에는 클래스 출력층이 없어 일반 Grad-CAM 적용이 불가능하다. FAISS에 저장된 매칭 임베딩과의 코사인 유사도를 역전파 타겟으로 사용하는 방식으로 변형했다.

### 다중 사진 등록 평균 (최종 채택)
강아지 한 마리당 여러 장의 사진을 등록하고 임베딩을 평균하는 방식으로, 단일 사진 대비 Rank-1이 86.0%→90.5%, Rank-5가 95.5%→98.5%로 개선됨을 확인했다.

---

## 최종 성능

| 지표 | 값 |
|---|---|
| YOLOv8n mAP50 | 0.995 |
| TAR (1:1 검증) | 57.8% |
| FAR @ threshold 0.4 | 3.65% |
| Rank-1 (다중 사진 평균) | 90.5% |
| Rank-5 (다중 사진 평균) | 98.5% |

---

## GitHub 저장소

전체 소스코드, 학습된 모델 가중치 7종, 데모용 등록 DB(50마리)를 공개 배포함.

https://github.com/jangyerinn/PawID

---

## 참고문헌

1. Hermans, A., Beyer, L., & Bastian, B. (2017). In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.
2. Tan, M., & Le, Q. V. (2019). EfficientNet: Rethinking Model Scaling for CNNs. ICML.
3. Schroff, F., et al. (2015). FaceNet: A Unified Embedding for Face Recognition and Clustering. CVPR.
4. Johnson, J., et al. (2019). Billion-scale similarity search with GPUs (FAISS). IEEE Transactions on Big Data.
5. Selvaraju, R. R., et al. (2017). Grad-CAM. ICCV.
6. Jocher, G., et al. (2023). YOLO by Ultralytics (YOLOv8).
7. Müller, S., et al. (2019). Dog Nose Biometrics: A Systematic Review of Pattern Uniqueness.
