"""
실제 등록된 DB 기준 정확도 자동 검증
- bulk_register_val.py로 등록한 각 강아지의 source_pair(imageA)를 가져와서
- app.py와 동일한 파이프라인(YOLO+CLAHE+160+FAISS)으로 검색
- 검색 결과가 "자기 자신의 짝(imageA)"을 정확히 찾았는지 자동 채점

즉, "진짜 같은 강아지인지" 를 CVPR 데이터셋의 정답 레이블로 직접 확인하는 스크립트

실행:
    python test_identify_accuracy.py
"""

import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, cv2, json
from pathlib import Path
from torchvision import transforms
import timm
try:
    from tqdm import tqdm
except:
    def tqdm(x, **k): return x

MODEL_PATH = "D:/pawId/models/semi_hard_texture.pt"
YOLO_PATH  = "D:/pawId/runs/detect/nose_detector_v1/weights/best.pt"
DB_INDEX   = "D:/pawId/db/faiss_texture.index"
DB_META    = "D:/pawId/db/metadata_texture.json"
VAL_IMG    = "D:/pawId/pet_biometric_challenge_2022/validation/images"
IMG_SIZE   = 160
THRESHOLD  = 0.45


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
    if yolo is None: return img_bgr, None
    try:
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        results = yolo.predict(img_rgb, conf=0.25, verbose=False)
        boxes = results[0].boxes
        if len(boxes) == 0: return img_bgr, None
        best = boxes.conf.argmax()
        x1,y1,x2,y2 = map(int, boxes.xyxy[best].cpu().numpy())
        h,w = img_bgr.shape[:2]; pad=15
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
    return e / (np.linalg.norm(e) + 1e-8)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = BiometricModel(512)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=False))
    model = model.to(device); model.eval()
    tf = get_transform()

    yolo = None
    try:
        from ultralytics import YOLO
        if Path(YOLO_PATH).exists():
            yolo = YOLO(YOLO_PATH)
    except Exception:
        pass

    import faiss
    index = faiss.read_index(DB_INDEX)
    with open(DB_META, encoding="utf-8") as f:
        metadata = json.load(f)

    # source_pair가 있는 항목만 검증 대상 (bulk_register_val.py로 등록된 것들)
    testable = [m for m in metadata if "source_pair" in m]
    print("검증 대상:", len(testable), "마리 (source_pair 있는 항목)")
    print("DB 전체 등록 수:", index.ntotal)

    correct = 0
    rank5_correct = 0
    below_threshold = 0
    wrong = []

    val_dir = Path(VAL_IMG)

    for m in tqdm(testable, desc="검증 중"):
        true_dog_id = m["dog_id"]
        a_name = m["source_pair"]["imageA"]
        img_path = val_dir / a_name
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        cropped, bbox = img, None  # CVPR 데이터셋은 이미 코 크롭됨 -> YOLO 재크롭 생략
        emb = get_embedding(model, cropped, tf, device).reshape(1, -1).astype(np.float32)
        faiss.normalize_L2(emb)

        k = min(5, index.ntotal)
        sims, idxs = index.search(emb, k)
        top1_sim, top1_idx = float(sims[0][0]), int(idxs[0][0])
        predicted_dog_id = metadata[top1_idx]["dog_id"]

        retrieved_ids = [metadata[int(idx)]["dog_id"] for idx in idxs[0]]

        is_correct = (predicted_dog_id == true_dog_id)
        if is_correct:
            correct += 1
        else:
            wrong.append({
                "true_dog_id": true_dog_id,
                "predicted_dog_id": predicted_dog_id,
                "imageA": a_name,
                "top1_sim": round(top1_sim, 3),
            })

        if true_dog_id in retrieved_ids:
            rank5_correct += 1

        if top1_sim < THRESHOLD:
            below_threshold += 1

    n = len(testable)
    print("\n" + "="*50)
    print("[실제 DB 기준 정확도 - 정답(source_pair) 대비 검증]")
    print("  Rank-1 정확도 :", round(correct/n*100, 2), "%  (", correct, "/", n, ")")
    print("  Rank-5 정확도 :", round(rank5_correct/n*100, 2), "%")
    print("  Threshold(", THRESHOLD, ") 미달로 '확신 부족' 처리된 수:", below_threshold)
    print("="*50)

    if wrong:
        print("\n[틀린 케이스 예시 - 최대 10개]")
        for w in wrong[:10]:
            print(" ", w)


if __name__ == "__main__":
    main()
