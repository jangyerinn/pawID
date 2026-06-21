"""
앙상블 테스트 (추가 학습 없음)
semi_hard.pt(224px, 원본) + semi_hard_texture.pt(160px, CLAHE) 임베딩을 결합
-> Rank-1/5 정확도가 단일 모델보다 좋아지는지 확인

실행:
    python ensemble_test.py --n 50
"""
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, cv2, json, csv, argparse
from pathlib import Path
from torchvision import transforms
import timm
try:
    from tqdm import tqdm
except:
    def tqdm(x, **k): return x

MODEL_A_PATH = "D:/pawId/models/semi_hard.pt"           # 224px, 원본
MODEL_B_PATH = "D:/pawId/models/semi_hard_texture.pt"   # 160px, CLAHE
VAL_CSV      = "D:/pawId/pet_biometric_challenge_2022/validation/valid_data.csv"
VAL_IMG      = "D:/pawId/pet_biometric_challenge_2022/validation/images"


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


def tf_A():  # semi_hard.pt: 224px, CLAHE 없음
    return transforms.Compose([
        transforms.ToPILImage(), transforms.Resize((224,224)), transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])

def tf_B():  # semi_hard_texture.pt: 160px, CLAHE 적용
    return transforms.Compose([
        transforms.ToPILImage(), transforms.Resize((160,160)), transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])


def embed_A(model, img_bgr, tf, device):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    t = tf(img_rgb).unsqueeze(0).to(device)
    with torch.no_grad():
        e = model(t).cpu().numpy()[0]
    return e / (np.linalg.norm(e) + 1e-8)

def embed_B(model, img_bgr, tf, device):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_rgb = apply_clahe(img_rgb)
    t = tf(img_rgb).unsqueeze(0).to(device)
    with torch.no_grad():
        e = model(t).cpu().numpy()[0]
    return e / (np.linalg.norm(e) + 1e-8)


def ensemble_embed(modelA, modelB, img_bgr, tfA, tfB, device):
    eA = embed_A(modelA, img_bgr, tfA, device)   # (512,)
    eB = embed_B(modelB, img_bgr, tfB, device)   # (512,)
    combined = np.concatenate([eA, eB])           # (1024,)
    return combined / (np.linalg.norm(combined) + 1e-8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    device = args.device if torch.cuda.is_available() else "cpu"

    modelA = BiometricModel(512); modelA.load_state_dict(torch.load(MODEL_A_PATH, map_location=device, weights_only=False))
    modelA = modelA.to(device); modelA.eval()
    modelB = BiometricModel(512); modelB.load_state_dict(torch.load(MODEL_B_PATH, map_location=device, weights_only=False))
    modelB = modelB.to(device); modelB.eval()
    tfA, tfB = tf_A(), tf_B()
    print("두 모델 로드 완료, device:", device)

    pairs = []
    with open(VAL_CSV, newline="", encoding="utf-8") as f:
        reader = csv.reader(f); next(reader)
        for row in reader:
            if len(row) >= 2:
                pairs.append((row[0].strip().replace("*","_"), row[1].strip().replace("*","_")))
    pairs = pairs[:args.n]
    val_dir = Path(VAL_IMG)

    # 갤러리 구축 (imageB)
    gallery_embs_single  = []   # semi_hard_texture만 (기존 비교용)
    gallery_embs_ensemble = []
    valid_pairs = []

    for a_name, b_name in tqdm(pairs, desc="갤러리 구축"):
        imgB = cv2.imread(str(val_dir / b_name))
        if imgB is None: continue
        gallery_embs_single.append(embed_B(modelB, imgB, tfB, device))
        gallery_embs_ensemble.append(ensemble_embed(modelA, modelB, imgB, tfA, tfB, device))
        valid_pairs.append((a_name, b_name))

    gallery_single   = np.array(gallery_embs_single)
    gallery_ensemble = np.array(gallery_embs_ensemble)

    # 쿼리 (imageA) 검색 - 단일모델 vs 앙상블 비교
    correct_single = correct_ensemble = 0
    rank5_single = rank5_ensemble = 0

    for i, (a_name, b_name) in enumerate(tqdm(valid_pairs, desc="검색 중")):
        imgA = cv2.imread(str(val_dir / a_name))
        if imgA is None: continue

        # 단일 모델(texture)
        qA_single = embed_B(modelB, imgA, tfB, device)
        sims_single = gallery_single @ qA_single
        order_single = np.argsort(-sims_single)
        if order_single[0] == i: correct_single += 1
        if i in order_single[:5]: rank5_single += 1

        # 앙상블
        qA_ens = ensemble_embed(modelA, modelB, imgA, tfA, tfB, device)
        sims_ens = gallery_ensemble @ qA_ens
        order_ens = np.argsort(-sims_ens)
        if order_ens[0] == i: correct_ensemble += 1
        if i in order_ens[:5]: rank5_ensemble += 1

    n = len(valid_pairs)
    print("\n" + "="*50)
    print("[비교 결과] n =", n)
    print("  단일(texture)  Rank-1:", round(correct_single/n*100,2), "%   Rank-5:", round(rank5_single/n*100,2), "%")
    print("  앙상블(A+B)    Rank-1:", round(correct_ensemble/n*100,2), "%   Rank-5:", round(rank5_ensemble/n*100,2), "%")
    print("="*50)

if __name__ == "__main__":
    main()
