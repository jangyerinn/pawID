"""
다중 사진 등록 평균 테스트
- train 데이터(마리당 평균 3.3장) 사용
- 갤러리: 강아지당 여러 장의 임베딩을 평균 -> 더 안정적인 "대표 벡터"
- 단일 사진 갤러리 vs 다중 사진 평균 갤러리 Rank-1/5 비교

실행:
    python multi_photo_test.py --n_dogs 200
"""
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, cv2, random, argparse
from pathlib import Path
from torchvision import transforms
import timm
try:
    from tqdm import tqdm
except:
    def tqdm(x, **k): return x

MODEL_PATH = "D:/pawId/models/semi_hard_texture.pt"
TRAIN_ROOT = "D:/pawId/processed/train"


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


def get_transform():
    return transforms.Compose([transforms.ToPILImage(), transforms.Resize((160,160)),
            transforms.ToTensor(), transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])


def embed(model, img_bgr, tf, device):
    rgb = apply_clahe(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    t = tf(rgb).unsqueeze(0).to(device)
    with torch.no_grad(): e = model(t).cpu().numpy()[0]
    return e / (np.linalg.norm(e)+1e-8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_dogs", type=int, default=200, help="테스트할 강아지 수")
    parser.add_argument("--gallery_imgs", type=int, default=2, help="갤러리당 등록 사진 수")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    device = args.device if torch.cuda.is_available() else "cpu"
    random.seed(args.seed)

    model = BiometricModel(512)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=False))
    model = model.to(device); model.eval()
    tf = get_transform()

    # gallery_imgs+1장 이상 가진 강아지만 선택 (등록용 N장 + 쿼리용 1장)
    need = args.gallery_imgs + 1
    dog_dirs = [d for d in Path(TRAIN_ROOT).iterdir() if d.is_dir()]
    valid_dogs = []
    for d in dog_dirs:
        imgs = list(d.glob("*.jpg"))
        if len(imgs) >= need:
            valid_dogs.append((d.name, imgs))
    print("조건(", need, "장 이상) 만족 강아지:", len(valid_dogs))

    if len(valid_dogs) > args.n_dogs:
        valid_dogs = random.sample(valid_dogs, args.n_dogs)
    n = len(valid_dogs)
    print("테스트 강아지 수:", n)

    gallery_single, gallery_multi, query_embs = [], [], []

    for dog_id, imgs in tqdm(valid_dogs, desc="임베딩 계산"):
        random.shuffle(imgs)
        gallery_paths = imgs[:args.gallery_imgs]
        query_path    = imgs[args.gallery_imgs]

        embs = []
        for p in gallery_paths:
            img = cv2.imread(str(p))
            if img is not None:
                embs.append(embed(model, img, tf, device))

        if not embs: continue

        gallery_single.append(embs[0])                                   # 1장만 사용
        multi = np.mean(embs, axis=0); multi = multi/(np.linalg.norm(multi)+1e-8)
        gallery_multi.append(multi)                                       # N장 평균

        qimg = cv2.imread(str(query_path))
        query_embs.append(embed(model, qimg, tf, device))

    gallery_single = np.array(gallery_single)
    gallery_multi  = np.array(gallery_multi)
    query_embs     = np.array(query_embs)

    sims_single = query_embs @ gallery_single.T
    sims_multi  = query_embs @ gallery_multi.T

    def rank_acc(sims):
        correct = rank5 = 0
        n_ = sims.shape[0]
        for i in range(n_):
            order = np.argsort(-sims[i])
            if order[0] == i: correct += 1
            if i in order[:5]: rank5 += 1
        return correct/n_*100, rank5/n_*100

    r1_s, r5_s = rank_acc(sims_single)
    r1_m, r5_m = rank_acc(sims_multi)

    print("\n" + "="*50)
    print("[다중 사진 평균 효과 검증] n =", len(gallery_single))
    print(f"  단일 사진 갤러리 (1장)        Rank-1={r1_s:.1f}%  Rank-5={r5_s:.1f}%")
    print(f"  다중 사진 평균 갤러리({args.gallery_imgs}장) Rank-1={r1_m:.1f}%  Rank-5={r5_m:.1f}%")
    print("="*50)

if __name__ == "__main__":
    main()
