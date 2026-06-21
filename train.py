"""
학습 전략: Triplet Loss + PK Sampler + Semi-Hard Negative Mining

- PK Sampler: 일반 랜덤 DataLoader는 배치 내 Positive 쌍이 우연히 들어갈
  확률이 낮아 Loss 계산 자체가 부실함. P마리 × K장으로 Positive 쌍 보장.
- Semi-Hard Negative Mining: d(A,P) < d(A,N) < d(A,P)+margin 구간에서
  Negative를 선택 → Easy(너무 쉬움)도, Hard(발산 위험)도 아닌 최적 구간.
  (Hermans et al., 2017, arXiv:1703.07737)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Sampler, Dataset, DataLoader
from collections import defaultdict
import random
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional


# ------------------------------------------------------------------
# 1. PK Sampler
# ------------------------------------------------------------------
class PKSampler(Sampler):
    """
    P마리 개체 × K장 이미지로 배치를 구성하는 샘플러
    → 배치 내 Positive 쌍(같은 강아지)을 항상 보장

    Args:
        labels:     각 샘플의 개체 ID 레이블 리스트
        P:          배치당 개체(강아지) 수
        K:          개체당 이미지 수
        drop_last:  마지막 불완전 배치 제거 여부
    """

    def __init__(self, labels: List[int], P: int = 8, K: int = 4, drop_last: bool = True):
        super().__init__(None)
        self.P = P
        self.K = K
        self.drop_last = drop_last

        # 개체 ID → 샘플 인덱스 매핑
        self.id_to_indices: Dict[int, List[int]] = defaultdict(list)
        for idx, label in enumerate(labels):
            self.id_to_indices[label].append(idx)

        # K장 이상인 개체만 사용
        self.valid_ids = [
            dog_id for dog_id, indices in self.id_to_indices.items()
            if len(indices) >= K
        ]

        if len(self.valid_ids) < P:
            raise ValueError(
                f"K={K}장 이상인 개체가 {len(self.valid_ids)}마리뿐 (P={P} 필요)"
            )

    def __iter__(self):
        all_indices = []
        ids = self.valid_ids.copy()
        random.shuffle(ids)

        for i in range(0, len(ids) - self.P + 1, self.P):
            batch_ids = ids[i: i + self.P]
            batch = []
            for dog_id in batch_ids:
                sampled = random.choices(self.id_to_indices[dog_id], k=self.K)
                batch.extend(sampled)
            all_indices.extend(batch)

        return iter(all_indices)

    def __len__(self):
        n_batches = len(self.valid_ids) // self.P
        return n_batches * self.P * self.K


# ------------------------------------------------------------------
# 2. 비문 데이터셋
# ------------------------------------------------------------------
class NoseDataset(Dataset):
    """
    CVPR 2022 Pet Biometric Challenge 데이터셋 래퍼

    디렉토리 구조:
        data/
          train/
            dog_001/  ← 개체 ID 폴더명
              img1.jpg
              img2.jpg
            dog_002/
              ...

    Args:
        root:       데이터 루트 경로
        transform:  이미지 변환 (get_transform() 사용)
    """

    def __init__(self, root: str, transform=None):
        self.root = Path(root)
        self.transform = transform
        self.samples = []   # (이미지 경로, 레이블 int)
        self.labels = []    # PKSampler용 레이블 리스트

        label_map = {}
        for dog_dir in sorted(self.root.iterdir()):
            if not dog_dir.is_dir():
                continue
            label = label_map.setdefault(dog_dir.name, len(label_map))
            for img_path in dog_dir.glob("*.jpg"):
                self.samples.append((img_path, label))
                self.labels.append(label)
            for img_path in dog_dir.glob("*.png"):
                self.samples.append((img_path, label))
                self.labels.append(label)

        self.num_classes = len(label_map)
        print(f"[데이터셋] {self.num_classes}마리, {len(self.samples)}장 로드 완료")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        import cv2
        img_path, label = self.samples[idx]
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            image = self.transform(image)

        return image, label


# ------------------------------------------------------------------
# 3. Triplet Loss + Semi-Hard Negative Mining
# ------------------------------------------------------------------
class TripletLoss(nn.Module):
    """
    Triplet Loss with Semi-Hard Negative Mining
    L = max(d(A,P) - d(A,N) + margin, 0)

    Semi-Hard 조건: d(A,P) < d(A,N) < d(A,P) + margin
    - Easy Negative (d(A,N) > d(A,P)+margin): 너무 쉬워 학습 효과 없음
    - Hard Negative (d(A,N) < d(A,P)):        학습 불안정, 발산 위험
    - Semi-Hard: 가장 안정적이고 효과적

    Ref: Hermans et al. (2017), arXiv:1703.07737
    """

    def __init__(self, margin: float = 0.3):
        super().__init__()
        self.margin = margin

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeddings: (B, D) L2 정규화된 임베딩 벡터
            labels:     (B,) 개체 ID

        Returns:
            스칼라 손실값
        """
        # 코사인 거리 행렬 (L2 정규화 후 내적 = 코사인 유사도)
        # 거리 = 1 - 코사인 유사도
        sim_matrix = torch.mm(embeddings, embeddings.t())          # (B, B)
        dist_matrix = 1.0 - sim_matrix                              # 코사인 거리

        labels = labels.unsqueeze(1)
        pos_mask = labels.eq(labels.t())                            # 같은 개체 쌍
        neg_mask = ~pos_mask
        pos_mask.fill_diagonal_(False)                              # 자기 자신 제외

        loss_values = []
        B = embeddings.size(0)

        for i in range(B):
            pos_indices = pos_mask[i].nonzero(as_tuple=True)[0]
            neg_indices = neg_mask[i].nonzero(as_tuple=True)[0]

            if len(pos_indices) == 0 or len(neg_indices) == 0:
                continue

            # 가장 먼 Positive (hardest positive)
            d_ap = dist_matrix[i, pos_indices].max()

            # Semi-Hard Negative: d_ap < d_an < d_ap + margin
            d_an_all = dist_matrix[i, neg_indices]
            semi_hard_mask = (d_an_all > d_ap) & (d_an_all < d_ap + self.margin)

            if semi_hard_mask.sum() > 0:
                d_an = d_an_all[semi_hard_mask].min()
            else:
                # Semi-Hard가 없으면 가장 Hard Negative 사용 (fallback)
                d_an = d_an_all.min()

            loss = F.relu(d_ap - d_an + self.margin)
            loss_values.append(loss)

        if not loss_values:
            return torch.tensor(0.0, requires_grad=True, device=embeddings.device)

        return torch.stack(loss_values).mean()


# ------------------------------------------------------------------
# 4. 학습 루프
# ------------------------------------------------------------------
def train_one_epoch(model, loader, criterion, optimizer, device, epoch):
    model.train()
    total_loss = 0.0
    n_batches = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        embeddings = model(images)
        loss = criterion(embeddings, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    avg_loss = total_loss / max(n_batches, 1)
    print(f"[Epoch {epoch}] Triplet Loss: {avg_loss:.4f}")
    return avg_loss


def train(
    model,
    train_root: str,
    val_root: Optional[str] = None,
    epochs: int = 50,
    P: int = 8,
    K: int = 4,
    margin: float = 0.3,
    lr: float = 1e-4,
    device: str = "cpu",
    save_path: str = "models/best_model.pt",
):
    """
    전체 학습 파이프라인

    Args:
        model:      BiometricModel 인스턴스
        train_root: 학습 데이터 루트 (개체 ID별 폴더 구조)
        val_root:   검증 데이터 루트 (없으면 생략)
        epochs:     학습 에폭 수
        P, K:       PK Sampler 설정 (P마리 × K장)
        margin:     Triplet Loss margin
        lr:         학습률
        device:     'cpu' | 'cuda'
    """
    from model import get_transform

    # 데이터셋 및 DataLoader
    train_dataset = NoseDataset(train_root, transform=get_transform(train=True))
    sampler = PKSampler(train_dataset.labels, P=P, K=K)
    train_loader = DataLoader(
        train_dataset,
        batch_size=P * K,
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
    )

    # 손실 함수 및 옵티마이저
    criterion = TripletLoss(margin=margin)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    model = model.to(device)
    best_loss = float("inf")

    for epoch in range(1, epochs + 1):
        loss = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)
        scheduler.step()

        if loss < best_loss:
            best_loss = loss
            model.save(save_path)
            print(f"  → 최적 모델 저장 (loss: {best_loss:.4f})")

    print(f"\n[학습 완료] 최적 loss: {best_loss:.4f}, 모델: {save_path}")
    return model
