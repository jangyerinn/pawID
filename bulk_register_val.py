"""
valid_data.csv (imageA, imageB) 쌍을 한 번에 DB에 등록
- 각 쌍의 imageB(고화질)를 가상의 강아지로 등록
- 나중에 imageA(저화질)로 식별 테스트 가능
- 텍스처 모델(semi_hard_texture.pt) + CLAHE + 160px 기준으로 등록 (app.py와 동일)

실행:
    python bulk_register_val.py --n 50      # 50쌍만 등록 (빠른 테스트)
    python bulk_register_val.py --n 2000    # 전체 등록 (시간 더 걸림)
"""

import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, cv2, faiss, json, csv, argparse
from pathlib import Path
from torchvision import transforms
import timm
try:
    from tqdm import tqdm
except:
    def tqdm(x, **k): return x

MODEL_PATH = "D:/pawId/models/semi_hard_texture.pt"
YOLO_PATH  = "D:/pawId/runs/detect/nose_detector_v1/weights/best.pt"
VAL_CSV    = "D:/pawId/pet_biometric_challenge_2022/validation/valid_data.csv"
VAL_IMG    = "D:/pawId/pet_biometric_challenge_2022/validation/images"
DB_INDEX   = "D:/pawId/db/faiss_texture.index"
DB_META    = "D:/pawId/db/metadata_texture.json"
DB_IMG_DIR = "D:/pawId/db/images_texture"
IMG_SIZE   = 160


def apply_clahe(img_rgb):
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l2 = clahe.apply(l)
    lab2 = cv2.merge([l2, a, b])
    return cv2.cvtColor(lab2, cv2.COLOR_LAB2RGB)


class EmbeddingHead(nn.Module):
    def __init__(self, in_features=1280, embed_dim=512, dropout=0.3):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(in_features, 512), nn.BatchNorm1d(512),
            nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(512, embed_dim),
        )
    def forward(self, x):
        return F.normalize(self.head(x), p=2, dim=1)


class BiometricModel(nn.Module):
    def __init__(self, embed_dim=512):
        super().__init__()
        self.backbone = timm.create_model("efficientnet_b0", pretrained=False, num_classes=0, global_pool="avg")
        self.embedding_head = EmbeddingHead(in_features=self.backbone.num_features, embed_dim=embed_dim)
    def forward(self, x):
        return self.embedding_head(self.backbone(x))


def get_transform(img_size=IMG_SIZE):
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])


def detect_and_crop(yolo, img_bgr):
    if yolo is None:
        return img_bgr, None
    try:
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        results = yolo.predict(img_rgb, conf=0.25, verbose=False)
        boxes = results[0].boxes
        if len(boxes) == 0:
            return img_bgr, None
        best = boxes.conf.argmax()
        x1,y1,x2,y2 = map(int, boxes.xyxy[best].cpu().numpy())
        h,w = img_bgr.shape[:2]
        pad = 15
        x1,y1 = max(0,x1-pad), max(0,y1-pad)
        x2,y2 = min(w,x2+pad), min(h,y2+pad)
        return img_bgr[y1:y2, x1:x2], (x1,y1,x2,y2)
    except Exception:
        return img_bgr, None


def get_embedding(model, img_bgr, tf, device):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_rgb = apply_clahe(img_rgb)
    t = tf(img_rgb).unsqueeze(0).to(device)
    with torch.no_grad():
        e = model(t).cpu().numpy().astype(np.float32)
    faiss.normalize_L2(e)
    return e


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=50, help="등록할 쌍 수 (기본 50, 전체는 2000)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--start_id", type=int, default=1, help="강아지 이름 시작 번호")
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    print("device:", device)

    # 모델 로드
    model = BiometricModel(512)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=False))
    model = model.to(device); model.eval()
    tf = get_transform()

    # YOLO 로드 (있으면 사용)
    yolo = None
    try:
        from ultralytics import YOLO
        if Path(YOLO_PATH).exists():
            yolo = YOLO(YOLO_PATH)
            print("YOLOv8 로드 완료")
    except Exception as e:
        print("YOLOv8 미사용:", e)

    # 기존 DB 로드 (있으면 이어서 추가)
    if Path(DB_INDEX).exists() and Path(DB_META).exists():
        index = faiss.read_index(DB_INDEX)
        with open(DB_META, encoding="utf-8") as f:
            metadata = json.load(f)
        print("기존 DB 로드:", index.ntotal, "마리")
    else:
        index = faiss.IndexFlatIP(512)
        metadata = []

    Path(DB_IMG_DIR).mkdir(parents=True, exist_ok=True)

    # CSV 파싱
    pairs = []
    with open(VAL_CSV, newline="", encoding="utf-8") as f:
        reader = csv.reader(f); next(reader)
        for row in reader:
            if len(row) >= 2:
                a = row[0].strip().replace("*","_")
                b = row[1].strip().replace("*","_")
                pairs.append((a, b))

    pairs = pairs[:args.n]
    print("등록할 쌍 수:", len(pairs))

    val_dir = Path(VAL_IMG)
    registered = 0
    skipped = 0

    for i, (a_name, b_name) in enumerate(tqdm(pairs, desc="등록 중")):
        # imageB(보통 더 고화질)를 등록용으로 사용
        img_path = val_dir / b_name
        img = cv2.imread(str(img_path))
        if img is None:
            skipped += 1
            continue

        cropped, bbox = img, None  # CVPR 데이터셋은 이미 코 크롭됨 -> YOLO 재크롭 생략
        emb = get_embedding(model, cropped, tf, device)

        new_id = index.ntotal
        index.add(emb)

        img_save_path = str(Path(DB_IMG_DIR) / (str(new_id) + ".jpg"))
        cv2.imwrite(img_save_path, cropped)

        dog_num = args.start_id + i
        metadata.append({
            "dog_id":   str(new_id),
            "dog_name": "dog_" + str(dog_num).zfill(4),
            "owner":    "테스트견주" + str(dog_num),
            "phone":    "010-0000-" + str(dog_num).zfill(4),
            "breed":    "미입력",
            "img_path": img_save_path,
            "source_pair": {"imageA": a_name, "imageB": b_name},  # 테스트용 - imageA로 검색 시 이 쌍을 찾으면 정답
        })
        registered += 1

    # 저장
    Path(DB_INDEX).parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, DB_INDEX)
    with open(DB_META, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print("\n" + "="*50)
    print("등록 완료:", registered, "마리 (실패:", skipped, ")")
    print("DB 총 등록 수:", index.ntotal)
    print("\n테스트 방법:")
    print("  1. app.py 실행")
    print("  2. validation/images/ 에서 등록된 쌍의 imageA 파일 찾아서 업로드")
    print("     (metadata_texture.json의 'source_pair'->'imageA' 값 참고)")
    print("  3. 식별 탭에서 매칭되는지 확인")


if __name__ == "__main__":
    main()
