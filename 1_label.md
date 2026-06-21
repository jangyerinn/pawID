# 1주차: Roboflow 라벨링 가이드

## YOLOv8 코 탐지기용 dog_nose 라벨링

### 왜 라벨링이 필요한가?
기본 YOLOv8 COCO 클래스에는 **'dog nose' 클래스가 없음**.
'dog' 클래스만 있어서 그냥 실행하면 강아지 전신을 잡아버림 → 비문 크롭 불가.
직접 라벨링 후 파인튜닝 필수.

---

## 라벨링 절차

### 1. Roboflow 계정 생성
- https://roboflow.com → 무료 계정 생성

### 2. 프로젝트 생성
- New Project → Object Detection
- Project Name: `dog-nose-detector`
- Annotation Group: `dog_nose`

### 3. 이미지 수집 기준 (200장)
다양한 환경에서 촬영:
- [ ] 낮/밤/실내/실외 조명 다양하게
- [ ] 정면, 측면, 위에서 다양한 각도
- [ ] 여러 견종 (말티즈, 골든, 시바 등)
- [ ] IQA 기준 통과하는 선명한 사진만 사용
  - Laplacian Variance ≥ 100 (선명도)
  - 픽셀 평균값 ≥ 40 (밝기)

### 4. 라벨링 방법
- Upload → 이미지 업로드 (100~200장)
- Annotate → 각 이미지에서 코 영역에 바운딩 박스 그리기
- 클래스: `dog_nose` 1종만
- 박스: 코 끝부분의 비문 무늬가 포함되도록 (너무 좁으면 안 됨)

### 5. Export
- Generate Dataset → YOLOv8 형식으로 export
- `data.yaml` 파일 생성됨:
  ```yaml
  train: ../train/images
  val:   ../valid/images
  nc: 1
  names: ['dog_nose']
  ```

---

## IQA 기준으로 라벨링 이미지 필터링

```python
import cv2
from src.iqa import IQAModule

iqa = IQAModule(min_sharpness=100, min_brightness=40)

for img_path in image_paths:
    image = cv2.imread(img_path)
    result = iqa.check(image)
    if not result.passed:
        print(f"제거: {img_path} - {result.message}")
```

---

## 예상 소요 시간
- 이미지 수집: 1~2시간
- 라벨링 (200장): 2~3시간
- Export 및 구조 확인: 30분

**1주차 내 완료 목표!** (YOLOv8 파인튜닝은 2주차)
