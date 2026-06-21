"""
FAR (False Acceptance Rate) 측정
서로 다른 강아지 쌍의 유사도를 측정해서
threshold별로 "남의 개를 내 개로 착각하는 비율"을 계산
"""
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, cv2, random
from pathlib import Path
import timm
from torchvision import transforms

MODEL_PATH = "D:/pawId/models/semi_hard.pt"
TRAIN_ROOT = "D:/pawId/processed/train"
N_PAIRS    = 2000

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

def get_transform():
    return transforms.Compose([
        transforms.ToPILImage(), transforms.Resize((224,224)), transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])

def get_embedding(model, img_bgr, tf, device):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    t = tf(img_rgb).unsqueeze(0).to(device)
    with torch.no_grad():
        e = model(t).cpu().numpy()[0]
    return e / (np.linalg.norm(e) + 1e-8)

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = BiometricModel(512)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=False))
    model = model.to(device); model.eval()
    tf = get_transform()

    dogs = [d for d in Path(TRAIN_ROOT).iterdir() if d.is_dir()]
    print("총", len(dogs), "마리 중 무작위 impostor pair", N_PAIRS, "개 생성")

    random.seed(0)
    sims = []
    for _ in range(N_PAIRS):
        d1, d2 = random.sample(dogs, 2)   # 서로 다른 두 강아지
        imgs1 = list(d1.glob("*.jpg"))
        imgs2 = list(d2.glob("*.jpg"))
        if not imgs1 or not imgs2: continue
        img1 = cv2.imread(str(random.choice(imgs1)))
        img2 = cv2.imread(str(random.choice(imgs2)))
        if img1 is None or img2 is None: continue
        e1 = get_embedding(model, img1, tf, device)
        e2 = get_embedding(model, img2, tf, device)
        sims.append(float(np.dot(e1, e2)))

    sims = np.array(sims)
    print("\nImpostor(다른 개) 유사도 통계")
    print("  평균:", round(sims.mean(),4))
    print("  최대:", round(sims.max(),4))
    print("  90백분위:", round(np.percentile(sims,90),4))
    print("  95백분위:", round(np.percentile(sims,95),4))
    print("  99백분위:", round(np.percentile(sims,99),4))

    print("\nThreshold별 FAR (다른 개를 같은 개로 착각하는 비율)")
    for t in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
        far = (sims >= t).mean()
        print(f"  threshold={t}  FAR={far*100:.2f}%")

if __name__ == "__main__":
    main()
