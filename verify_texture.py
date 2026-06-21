"""
텍스처 강화 모델(semi_hard_texture.pt) 검증
- CLAHE + 160x160 전처리를 학습과 동일하게 적용
- FAR(다른 개 오인식) + TAR(같은 개 인식, val 쌍) 동시 측정
- 기존 semi_hard.pt 결과와 비교 가능

실행:
    python verify_texture.py
"""
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, cv2, random, csv
from pathlib import Path
import timm
from torchvision import transforms

MODEL_PATH = "D:/pawId/models/semi_hard_texture.pt"
TRAIN_ROOT = "D:/pawId/processed/train"
VAL_CSV    = "D:/pawId/pet_biometric_challenge_2022/validation/valid_data.csv"
VAL_IMG    = "D:/pawId/pet_biometric_challenge_2022/validation/images"
IMG_SIZE   = 160
N_PAIRS    = 2000


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


def get_eval_transform(img_size=IMG_SIZE):
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])


def get_embedding(model, img_bgr, tf, device):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_rgb = apply_clahe(img_rgb)          # 학습과 동일한 CLAHE 적용
    t = tf(img_rgb).unsqueeze(0).to(device)
    with torch.no_grad():
        e = model(t).cpu().numpy()[0]
    return e / (np.linalg.norm(e) + 1e-8)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL_PATH)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = BiometricModel(512)
    model.load_state_dict(torch.load(args.model, map_location=device, weights_only=False))
    model = model.to(device); model.eval()
    tf = get_eval_transform()
    print("모델 로드:", args.model, " device:", device)

    # ── 1. FAR (다른 개 오인식률) ──
    dogs = [d for d in Path(TRAIN_ROOT).iterdir() if d.is_dir()]
    random.seed(0)
    far_sims = []
    for _ in range(N_PAIRS):
        d1, d2 = random.sample(dogs, 2)
        imgs1, imgs2 = list(d1.glob("*.jpg")), list(d2.glob("*.jpg"))
        if not imgs1 or not imgs2: continue
        img1 = cv2.imread(str(random.choice(imgs1)))
        img2 = cv2.imread(str(random.choice(imgs2)))
        if img1 is None or img2 is None: continue
        e1 = get_embedding(model, img1, tf, device)
        e2 = get_embedding(model, img2, tf, device)
        far_sims.append(float(np.dot(e1, e2)))
    far_sims = np.array(far_sims)

    print("\n[FAR] Impostor(다른 개) 유사도 통계")
    print("  평균:", round(far_sims.mean(),4), " 최대:", round(far_sims.max(),4))
    print("  90백분위:", round(np.percentile(far_sims,90),4))
    print("  95백분위:", round(np.percentile(far_sims,95),4))
    print("  99백분위:", round(np.percentile(far_sims,99),4))
    print("\n  Threshold별 FAR")
    for t in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        print(f"    threshold={t}  FAR={(far_sims>=t).mean()*100:.2f}%")

    # ── 2. TAR (val genuine pair, 같은 개) ──
    pairs = []
    with open(VAL_CSV, newline="", encoding="utf-8") as f:
        reader = csv.reader(f); next(reader)
        for row in reader:
            if len(row) >= 2:
                a = row[0].strip().replace("*","_")
                b = row[1].strip().replace("*","_")
                pairs.append((a,b))

    val_dir = Path(VAL_IMG)
    tar_sims = []
    for a_name, b_name in pairs:
        imgA = cv2.imread(str(val_dir/a_name))
        imgB = cv2.imread(str(val_dir/b_name))
        if imgA is None or imgB is None: continue
        eA = get_embedding(model, imgA, tf, device)
        eB = get_embedding(model, imgB, tf, device)
        tar_sims.append(float(np.dot(eA,eB)))
    tar_sims = np.array(tar_sims)

    print("\n[TAR] Genuine(같은 개, 화질 다름) 유사도 통계")
    print("  평균:", round(tar_sims.mean(),4), " 최소:", round(tar_sims.min(),4))
    print("\n  Threshold별 TAR")
    for t in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        print(f"    threshold={t}  TAR={(tar_sims>=t).mean()*100:.2f}%")

    print("\n=== 비교 요약 (기존 semi_hard.pt 결과) ===")
    print("  semi_hard.pt : TAR@0.3=60.0%  FAR@0.3=12.25%  FAR@0.4=6.55%  최대impostor=0.7778")
    print("  texture 모델 : 위 결과와 비교해보세요")


if __name__ == "__main__":
    main()
