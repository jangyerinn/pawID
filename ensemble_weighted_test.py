"""
가중치 기반 앙상블 (벡터 concat 대신 유사도 점수를 가중합)
score = w * simB(texture) + (1-w) * simA(original)
w를 0~1 스캔해서 최적 가중치를 찾음

실행:
    python ensemble_weighted_test.py --n 50
"""
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, cv2, csv, argparse
from pathlib import Path
from torchvision import transforms
import timm
try:
    from tqdm import tqdm
except:
    def tqdm(x, **k): return x

MODEL_A_PATH = "D:/pawId/models/semi_hard.pt"
MODEL_B_PATH = "D:/pawId/models/semi_hard_texture.pt"
VAL_CSV      = "D:/pawId/pet_biometric_challenge_2022/validation/valid_data.csv"
VAL_IMG      = "D:/pawId/pet_biometric_challenge_2022/validation/images"


def apply_clahe(img_rgb):
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l2 = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l2, a, b]), cv2.COLOR_LAB2RGB)


class EmbeddingHead(nn.Module):
    def __init__(self, in_features=1280, embed_dim=512, dropout=0.3):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(in_features, 512), nn.BatchNorm1d(512),
            nn.ReLU(inplace=True), nn.Dropout(dropout), nn.Linear(512, embed_dim),
        )
    def forward(self, x): return F.normalize(self.head(x), p=2, dim=1)


class BiometricModel(nn.Module):
    def __init__(self, embed_dim=512):
        super().__init__()
        self.backbone = timm.create_model("efficientnet_b0", pretrained=False, num_classes=0, global_pool="avg")
        self.embedding_head = EmbeddingHead(in_features=self.backbone.num_features, embed_dim=embed_dim)
    def forward(self, x): return self.embedding_head(self.backbone(x))


def tf_A(): return transforms.Compose([transforms.ToPILImage(), transforms.Resize((224,224)),
            transforms.ToTensor(), transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
def tf_B(): return transforms.Compose([transforms.ToPILImage(), transforms.Resize((160,160)),
            transforms.ToTensor(), transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])

def embed_A(model, img_bgr, tf, device):
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    t = tf(rgb).unsqueeze(0).to(device)
    with torch.no_grad(): e = model(t).cpu().numpy()[0]
    return e / (np.linalg.norm(e)+1e-8)

def embed_B(model, img_bgr, tf, device):
    rgb = apply_clahe(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    t = tf(rgb).unsqueeze(0).to(device)
    with torch.no_grad(): e = model(t).cpu().numpy()[0]
    return e / (np.linalg.norm(e)+1e-8)


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

    pairs = []
    with open(VAL_CSV, newline="", encoding="utf-8") as f:
        r = csv.reader(f); next(r)
        for row in r:
            if len(row) >= 2:
                pairs.append((row[0].strip().replace("*","_"), row[1].strip().replace("*","_")))
    pairs = pairs[:args.n]
    val_dir = Path(VAL_IMG)

    # 갤러리(imageB) - 두 모델 임베딩 각각 미리 계산
    gA, gB, valid = [], [], []
    for a_name, b_name in tqdm(pairs, desc="갤러리 구축"):
        img = cv2.imread(str(val_dir / b_name))
        if img is None: continue
        gA.append(embed_A(modelA, img, tfA, device))
        gB.append(embed_B(modelB, img, tfB, device))
        valid.append((a_name, b_name))
    gA = np.array(gA); gB = np.array(gB)

    # 쿼리(imageA) - 두 모델 임베딩 각각 미리 계산, 유사도 행렬도 미리 계산
    qA_list, qB_list = [], []
    for a_name, b_name in tqdm(valid, desc="쿼리 임베딩"):
        img = cv2.imread(str(val_dir / a_name))
        qA_list.append(embed_A(modelA, img, tfA, device))
        qB_list.append(embed_B(modelB, img, tfB, device))
    qA_arr = np.array(qA_list); qB_arr = np.array(qB_list)

    simA_matrix = qA_arr @ gA.T   # (N, N)
    simB_matrix = qB_arr @ gB.T   # (N, N)

    n = len(valid)
    print("\n가중치별 Rank-1 / Rank-5 (w=texture 비중)")
    print(f"{'w':>5} {'Rank-1':>8} {'Rank-5':>8}")

    best_w, best_r1 = 0.0, -1
    for w in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        combined = w * simB_matrix + (1-w) * simA_matrix
        correct = rank5 = 0
        for i in range(n):
            order = np.argsort(-combined[i])
            if order[0] == i: correct += 1
            if i in order[:5]: rank5 += 1
        r1, r5 = correct/n*100, rank5/n*100
        print(f"{w:>5.1f} {r1:>7.1f}% {r5:>7.1f}%")
        if r1 > best_r1:
            best_r1, best_w = r1, w

    print(f"\n최적 가중치: w={best_w} (texture 비중)  ->  Rank-1={best_r1:.1f}%")
    print("참고: w=0.0은 순수 semi_hard.pt, w=1.0은 순수 semi_hard_texture.pt")


if __name__ == "__main__":
    main()
