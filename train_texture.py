"""
텍스처(비문 무늬) 강화 재학습
- CLAHE로 국소 대비 강화 -> 미세 텍스처 패턴 부각
- 입력 크기 224 -> 160 으로 축소 -> 모델이 큰 형태/색 대신 세밀한 패턴에 의존하도록 강제
- 나머지(Triplet + PK Sampler + Semi-Hard)는 기존 semi_hard와 동일하게 유지

실행:
    python train_texture.py --epochs 150
"""

import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, cv2, argparse, random
from pathlib import Path
from collections import defaultdict
from torch.utils.data import Dataset, Sampler, DataLoader
from torchvision import transforms
import timm
try:
    from tqdm import tqdm
except:
    def tqdm(x, **k): return x


# --------------------------------------------------
# CLAHE 텍스처 강화 전처리 (핵심 변경)
# --------------------------------------------------
def apply_clahe(img_rgb):
    """RGB 이미지에 CLAHE를 적용해 국소 대비(=미세 텍스처)를 강화"""
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l2 = clahe.apply(l)
    lab2 = cv2.merge([l2, a, b])
    return cv2.cvtColor(lab2, cv2.COLOR_LAB2RGB)


class CLAHETransform:
    """torchvision transform 파이프라인에 끼워 넣을 CLAHE 단계"""
    def __call__(self, img_rgb_np):
        return apply_clahe(img_rgb_np)


def get_texture_transform(train=True, img_size=160):
    steps = [CLAHETransform(), transforms.ToPILImage()]
    if train:
        steps += [
            transforms.RandomResizedCrop(img_size, scale=(0.6, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=20),
            transforms.ColorJitter(brightness=0.3, contrast=0.3),
        ]
    else:
        steps += [transforms.Resize((img_size, img_size))]
    steps += [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
    return transforms.Compose(steps)


# --------------------------------------------------
# 모델 (기존 semi_hard와 동일 구조)
# --------------------------------------------------
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
        self.backbone = timm.create_model("efficientnet_b0", pretrained=True, num_classes=0, global_pool="avg")
        self.embedding_head = EmbeddingHead(in_features=self.backbone.num_features, embed_dim=embed_dim)
    def forward(self, x):
        return self.embedding_head(self.backbone(x))


# --------------------------------------------------
# 데이터셋 / PK Sampler / Triplet Loss (기존과 동일)
# --------------------------------------------------
class NoseDataset(Dataset):
    def __init__(self, root, transform=None):
        self.transform = transform
        self.samples = []
        self.labels  = []
        label_map = {}
        for dog_dir in sorted(Path(root).iterdir()):
            if not dog_dir.is_dir(): continue
            label = label_map.setdefault(dog_dir.name, len(label_map))
            for p in list(dog_dir.glob("*.jpg")) + list(dog_dir.glob("*.png")):
                self.samples.append(str(p)); self.labels.append(label)
        self.num_classes = len(label_map)
        print("데이터셋:", self.num_classes, "마리,", len(self.samples), "장")

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        img = cv2.imread(self.samples[idx])
        if img is None:
            img = np.zeros((224,224,3), dtype=np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform: img = self.transform(img)
        return img, self.labels[idx]


class PKSampler(Sampler):
    def __init__(self, labels, P=8, K=4):
        super().__init__(None)
        self.P, self.K = P, K
        self.id_to_idx = defaultdict(list)
        for i, l in enumerate(labels): self.id_to_idx[l].append(i)
        self.valid_ids = [k for k,v in self.id_to_idx.items() if len(v) >= K]
        if len(self.valid_ids) < P:
            raise ValueError("K=" + str(K) + "장 이상 개체가 " + str(len(self.valid_ids)) + "마리뿐")

    def __iter__(self):
        ids = self.valid_ids.copy(); random.shuffle(ids)
        out = []
        for i in range(0, len(ids)-self.P+1, self.P):
            for dog_id in ids[i:i+self.P]:
                out.extend(random.choices(self.id_to_idx[dog_id], k=self.K))
        return iter(out)

    def __len__(self):
        return (len(self.valid_ids)//self.P) * self.P * self.K


class TripletLoss(nn.Module):
    def __init__(self, margin=0.5):
        super().__init__(); self.margin = margin

    def forward(self, embeddings, labels):
        sim  = torch.mm(embeddings, embeddings.t())
        dist = 1.0 - sim
        labels = labels.unsqueeze(1)
        pos_mask = labels.eq(labels.t()); pos_mask.fill_diagonal_(False)
        neg_mask = ~labels.eq(labels.t())
        losses = []
        for i in range(embeddings.size(0)):
            pos_idx = pos_mask[i].nonzero(as_tuple=True)[0]
            neg_idx = neg_mask[i].nonzero(as_tuple=True)[0]
            if len(pos_idx)==0 or len(neg_idx)==0: continue
            d_ap = dist[i, pos_idx].max()
            d_an_all = dist[i, neg_idx]
            semi = (d_an_all > d_ap) & (d_an_all < d_ap + self.margin)
            d_an = d_an_all[semi].min() if semi.sum()>0 else d_an_all.min()
            losses.append(F.relu(d_ap - d_an + self.margin))
        if not losses:
            return torch.tensor(0.0, requires_grad=True, device=embeddings.device)
        return torch.stack(losses).mean()


# --------------------------------------------------
# 학습 루프
# --------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_root", default="D:/pawId/processed/train")
    parser.add_argument("--model_path", default="D:/pawId/models/semi_hard_texture.pt")
    parser.add_argument("--epochs",     type=int, default=150)
    parser.add_argument("--img_size",   type=int, default=160)   # 224 -> 160
    parser.add_argument("--P", type=int, default=16)
    parser.add_argument("--K", type=int, default=2)
    parser.add_argument("--margin", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    print("=" * 50)
    print("텍스처 강화 재학습 (CLAHE + img_size=" + str(args.img_size) + ")")
    print("=" * 50)

    transform = get_texture_transform(train=True, img_size=args.img_size)
    dataset = NoseDataset(args.train_root, transform=transform)
    sampler = PKSampler(dataset.labels, P=args.P, K=args.K)
    loader  = DataLoader(dataset, batch_size=args.P*args.K, sampler=sampler, num_workers=4, pin_memory=True)

    model = BiometricModel(embed_dim=512).to(args.device)
    criterion = TripletLoss(margin=args.margin)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    Path(args.model_path).parent.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")

    for epoch in range(1, args.epochs+1):
        model.train()
        total, n = 0.0, 0
        for images, labels in loader:
            images, labels = images.to(args.device), labels.to(args.device)
            optimizer.zero_grad()
            emb = model(images)
            loss = criterion(emb, labels)
            loss.backward()
            optimizer.step()
            total += loss.item(); n += 1
        scheduler.step()
        avg = total / max(n,1)
        if avg < best_loss:
            best_loss = avg
            torch.save(model.state_dict(), args.model_path)
            mark = "  <- 저장"
        else:
            mark = ""
        if epoch % 5 == 0 or epoch == 1:
            print("[Epoch " + str(epoch) + "] loss=" + str(round(avg,4)) + mark)

    print("\n학습 완료! 최적 loss:", round(best_loss,4))
    print("모델 저장:", args.model_path)
    print("\n다음 단계: build_db.py / app.py 에 동일한 CLAHE+img_size 전처리 적용 필요")


if __name__ == "__main__":
    main()
