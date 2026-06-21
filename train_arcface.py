"""
ArcFace Loss 기반 재학습
- Triplet Loss: 배치 32개 안 negative만 비교 (전체의 0.5%)
- ArcFace:     6000마리 전체를 동시에 보면서 학습 (훨씬 강력)
- 마리당 3.3장처럼 데이터 적을 때 ArcFace가 압도적으로 유리

Ref: Deng et al. (2019). ArcFace: Additive Angular Margin Loss. CVPR.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import argparse
import math
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm
try:
    from tqdm import tqdm
except:
    def tqdm(x, **k): return x


# --------------------------------------------------
# 모델 정의
# --------------------------------------------------
class EmbeddingHead(nn.Module):
    def __init__(self, in_features=1280, embed_dim=512, dropout=0.3):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, embed_dim),
            nn.BatchNorm1d(embed_dim),  # ArcFace는 마지막 BN 중요
        )
    def forward(self, x):
        return self.head(x)  # ArcFace는 L2 정규화를 loss 내부에서 처리


class BiometricModel(nn.Module):
    def __init__(self, embed_dim=512):
        super().__init__()
        self.backbone = timm.create_model(
            "efficientnet_b0", pretrained=True,
            num_classes=0, global_pool="avg",
        )
        self.embedding_head = EmbeddingHead(
            in_features=self.backbone.num_features,
            embed_dim=embed_dim,
        )
    def forward(self, x):
        features = self.backbone(x)
        return self.embedding_head(features)

    def get_normalized_embedding(self, x):
        emb = self.forward(x)
        return F.normalize(emb, p=2, dim=1)


# --------------------------------------------------
# ArcFace Loss
# --------------------------------------------------
class ArcFaceLoss(nn.Module):
    """
    ArcFace: Additive Angular Margin Loss
    - 6000마리 전체를 분류 학습하면서 동시에 임베딩 공간 정렬
    - s=64.0: feature scale (임베딩 반지름)
    - m=0.5:  angular margin (클래스 간 간격 강제)
    """
    def __init__(self, embed_dim=512, num_classes=6000, s=64.0, m=0.5):
        super().__init__()
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embed_dim))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)   # threshold
        self.mm = math.sin(math.pi - m) * m

    def forward(self, embeddings, labels):
        # L2 정규화
        emb_norm = F.normalize(embeddings, p=2, dim=1)
        w_norm   = F.normalize(self.weight, p=2, dim=1)

        # 코사인 유사도
        cosine = F.linear(emb_norm, w_norm)           # (B, num_classes)
        sine   = torch.sqrt(1.0 - cosine.pow(2).clamp(0, 1))

        # cos(theta + m) = cos(theta)*cos(m) - sin(theta)*sin(m)
        phi = cosine * self.cos_m - sine * self.sin_m

        # 안정성: theta + m > pi 이면 phi 대신 cosine - mm 사용
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        # 정답 클래스에만 margin 적용
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)
        output = one_hot * phi + (1 - one_hot) * cosine

        return F.cross_entropy(output * self.s, labels)


# --------------------------------------------------
# 데이터셋
# --------------------------------------------------
class NoseDataset(Dataset):
    def __init__(self, root, transform=None):
        self.transform = transform
        self.samples = []
        self.labels  = []
        label_map = {}
        root = Path(root)
        for dog_dir in sorted(root.iterdir()):
            if not dog_dir.is_dir():
                continue
            label = label_map.setdefault(dog_dir.name, len(label_map))
            for img_path in list(dog_dir.glob("*.jpg")) + list(dog_dir.glob("*.png")):
                self.samples.append(str(img_path))
                self.labels.append(label)
        self.num_classes = len(label_map)
        print("데이터셋: " + str(self.num_classes) + "마리, " + str(len(self.samples)) + "장")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img = cv2.imread(self.samples[idx])
        if img is None:
            img = np.zeros((224, 224, 3), dtype=np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]


def get_train_transform():
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),   # 더 강한 크롭
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=30),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.RandomApply([
        transforms.GaussianBlur(kernel_size=7, sigma=(1.0, 5.0))
        ], p=0.5),
        transforms.RandomAutocontrast(p=0.3),
        transforms.RandomAdjustSharpness(sharpness_factor=0, p=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.1)),
    ])


# --------------------------------------------------
# 학습 루프
# --------------------------------------------------
def train(train_root, model_path, epochs, device, batch_size, lr, embed_dim=512):
    print("=" * 50)
    print("ArcFace 학습 시작")
    print("  epochs     : " + str(epochs))
    print("  batch_size : " + str(batch_size))
    print("  lr         : " + str(lr))
    print("  device     : " + device)
    print("=" * 50)

    dataset = NoseDataset(train_root, transform=get_train_transform())
    loader  = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True,
    )

    model      = BiometricModel(embed_dim=embed_dim).to(device)
    arcface    = ArcFaceLoss(
        embed_dim=embed_dim,
        num_classes=dataset.num_classes,
        s=64.0, m=0.5,
    ).to(device)

    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(arcface.parameters()),
        lr=lr, weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=50, T_mult=1, eta_min=1e-6,
    )

    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")

    for epoch in range(1, epochs + 1):
        model.train(); arcface.train()
        total_loss = 0.0
        n_batches  = 0

        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            embeddings = model(images)
            loss = arcface(embeddings, labels)
            loss.backward()

            # Gradient clipping (안정적인 학습)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches  += 1

        scheduler.step()
        avg_loss = total_loss / max(n_batches, 1)

        if avg_loss < best_loss:
            best_loss = avg_loss
            # 임베딩 모델만 저장 (ArcFace 헤드는 추론 시 불필요)
            torch.save(model.state_dict(), model_path)
            mark = "  <- 저장"
        else:
            mark = ""

        if epoch % 5 == 0 or epoch == 1:
            print("[Epoch " + str(epoch) + "] ArcFace Loss: " + str(round(avg_loss, 4)) + mark)

    print("\n학습 완료! 최적 loss: " + str(round(best_loss, 4)))
    print("모델 저장: " + model_path)
    print("다음 단계: python build_db.py --model " + model_path)
    return model_path


# --------------------------------------------------
# 메인
# --------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="ArcFace 재학습")
    parser.add_argument("--train_root",  default="D:/pawId/processed/train")
    parser.add_argument("--model_path",  default="D:/pawId/models/arcface.pt")
    parser.add_argument("--epochs",      type=int,   default=150)
    parser.add_argument("--batch_size",  type=int,   default=64)
    parser.add_argument("--lr",          type=float, default=1e-4)
    parser.add_argument("--embed_dim",   type=int,   default=512)
    parser.add_argument("--device",      default="cuda")
    args = parser.parse_args()

    train(
        train_root = args.train_root,
        model_path = args.model_path,
        epochs     = args.epochs,
        device     = args.device,
        batch_size = args.batch_size,
        lr         = args.lr,
        embed_dim  = args.embed_dim,
    )


if __name__ == "__main__":
    main()
