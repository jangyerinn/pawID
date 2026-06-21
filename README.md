# 🐾 PawID
**대조 학습 기반 유기견 비문(코 무늬) 생체 식별 시스템**

> Triplet Loss · IQA · Similarity-CAM을 활용한 유기견 신원 확인 시스템  
> 딥러닝실습 24012480 장예린

---

## 프로젝트 구조

```
PawID/
├── app.py               # Gradio 웹앱 (7주차)
├── requirements.txt     # 의존성 패키지
├── src/
│   ├── iqa.py           # IQA 모듈 (Laplacian Variance, 수학 알고리즘 기반)
│   ├── detector.py      # YOLOv8n 코 탐지기 (파인튜닝)
│   ├── model.py         # EfficientNet-B0 + Embedding Head
│   ├── train.py         # Triplet Loss + PK Sampler + Semi-Hard Negative Mining
│   ├── database.py      # FAISS 벡터 DB (등록 / 검색)
│   ├── cam.py           # Similarity-CAM (XAI, 코사인 유사도 역전파)
│   ├── evaluate.py      # Rank-1/5 Accuracy, EER, ROC Curve
│   └── pipeline.py      # 전체 시스템 통합 파이프라인
├── scripts/
│   ├── 1_label.md       # 1주차: Roboflow 라벨링 가이드
│   ├── 2_finetune.py    # 2주차: YOLOv8n 파인튜닝
│   ├── 3_train.py       # 3~4주차: Baseline → Triplet Loss 학습
│   └── 4_evaluate.py    # 7~8주차: ROC Curve, EER 분석
├── models/              # 학습된 가중치 저장 위치
└── data/                # FAISS 인덱스 및 메타데이터
```

---

## 개발 일정 (8주)

| 주차 | 작업 | 파일 |
|------|------|------|
| 1주차 | 환경 세팅, CVPR 데이터 다운로드, 코 라벨링 200장 (Roboflow), IQA 모듈 구현 | `src/iqa.py` |
| 2주차 | YOLOv8n 코 탐지기 파인튜닝, 코 크롭 파이프라인 구축 | `src/detector.py`, `scripts/2_finetune.py` |
| 3주차 | EfficientNet-B0 + Embedding Head 구현, Baseline 학습 | `src/model.py`, `scripts/3_train.py` |
| 4주차 | PK Sampler 구현, Triplet Loss + Semi-Hard Negative Mining 적용 | `src/train.py` |
| 5주차 | 데이터 증강, FAISS 인덱스 구축, IQA 파라미터 최적화 | `src/database.py` |
| 6주차 | Similarity-CAM 구현 (코사인 유사도 역전파 방식) | `src/cam.py` |
| 7주차 | 시연 DB 구축, Gradio 웹앱 완성 | `app.py` |
| 8주차 | 버그 수정, ROC Curve 분석, 최종 보고서 | `src/evaluate.py`, `scripts/4_evaluate.py` |

---

## 설치 및 실행

### 1. 환경 설치
```bash
pip install -r requirements.txt
```

### 2. 데이터 준비
- CVPR 2022 Pet Biometric Challenge Dataset: https://www.kaggle.com/datasets/zekunn/pet-biometric-challenge/data
- YOLOv8 라벨링: Roboflow에서 dog_nose 클래스 직접 라벨링 (200장)

데이터 구조:
```
data/
  train/
    dog_001/
      img1.jpg
      img2.jpg
    dog_002/
      ...
  val/
    ...
```

### 3. YOLOv8n 파인튜닝 (2주차)
```bash
python scripts/2_finetune.py
```

### 4. 비문 인식 모델 학습 (3~4주차)
```bash
# Baseline (ResNet50 + Contrastive Loss)
python scripts/3_train.py --mode baseline

# 실험 2: EfficientNet-B0 + Triplet Loss
python scripts/3_train.py --mode triplet

# 실험 3: + PK Sampler
python scripts/3_train.py --mode pk_sampler

# 실험 4: + Semi-Hard Negative Mining (최종)
python scripts/3_train.py --mode semi_hard
```

### 5. Gradio 앱 실행 (7주차)
```bash
# 로컬 실행
python app.py

# 데모 모드 (모델 없이)
python app.py --demo

# Hugging Face Spaces 공개 링크
python app.py --share
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

### Similarity-CAM
- 임베딩 모델에 일반 Grad-CAM 적용 시 에러 (클래스 출력층 없음)
- 코사인 유사도 점수를 역전파 타겟으로 사용하는 방식으로 변형
- "코의 이 부분 무늬 패턴이 일치해서 같은 강아지로 판단" 시각화

---

## 참고문헌

1. Hermans, A., Beyer, L., & Bastian, B. (2017). In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.
2. Tan, M., & Le, Q. V. (2019). EfficientNet: Rethinking Model Scaling for CNNs. ICML.
3. Schroff, F., et al. (2015). FaceNet: A Unified Embedding for Face Recognition and Clustering. CVPR.
4. Johnson, J., et al. (2019). Billion-scale similarity search with GPUs (FAISS). IEEE Transactions on Big Data.
5. Selvaraju, R. R., et al. (2017). Grad-CAM. ICCV.
6. Jocher, G., et al. (2023). YOLO by Ultralytics (YOLOv8).
7. Müller, S., et al. (2019). Dog Nose Biometrics: A Systematic Review of Pattern Uniqueness.
