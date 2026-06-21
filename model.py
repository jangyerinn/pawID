"""
Biometric Model: EfficientNet-B0 + Embedding Head
- EfficientNet-B0 (5.3M params): ResNet-50(25.6M)보다 가볍고 Compound Scaling으로
  미세한 비문 텍스처 패턴 포착에 유리
- Embedding Head: 1280D → 512D L2 정규화 벡터 (코사인 유사도 계산 용이)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
import cv2
from typing import Optional


# ------------------------------------------------------------------
# 전처리 파이프라인
# ------------------------------------------------------------------
def get_transform(train: bool = True):
    """
    EfficientNet-B0 입력 전처리 (224×224, ImageNet 통계)

    학습 시 데이터 증강:
    - 좌우/상하 반전, 회전 ±15도, 밝기/대비 ±0.3, 가우시안 블러
    - 색상 증강 배제: 비문은 텍스처 기반, 색상 왜곡 시 성능 저하
    """
    if train:
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.CenterCrop(200),
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.3, contrast=0.3),
            transforms.RandomApply([
             transforms.GaussianBlur(kernel_size=7, sigma=(1.0, 5.0))
            ], p=0.5),
            transforms.RandomAutocontrast(p=0.3),
            transforms.RandomAdjustSharpness(sharpness_factor=0, p=0.3),
            transforms.RandomCrop(224, padding=int(224 * 0.1)),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet 통계
                std=[0.229, 0.224, 0.225],
            ),
        ])
    else:
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])


# ------------------------------------------------------------------
# 모델 정의
# ------------------------------------------------------------------
class EmbeddingHead(nn.Module):
    """
    1280D → 512D 임베딩 압축 헤드
    1280 → Linear → BatchNorm → ReLU → Dropout(0.3) → Linear → 512 → L2 정규화
    """

    def __init__(self, in_features: int = 1280, embed_dim: int = 512, dropout: float = 0.3):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, embed_dim),
            nn.BatchNorm1d(embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.head(x)
        return F.normalize(x, p=2, dim=1)  # L2 정규화 → 단위 구 위에 임베딩


class BiometricModel(nn.Module):
    """
    EfficientNet-B0 Backbone + Embedding Head

    Args:
        embed_dim:  임베딩 차원 (기본 512)
        pretrained: ImageNet 사전학습 가중치 사용 여부
    """

    def __init__(self, embed_dim: int = 512, pretrained: bool = True):
        super().__init__()

        try:
            import timm
            self.backbone = timm.create_model(
                "efficientnet_b0",
                pretrained=pretrained,
                num_classes=0,       # 분류 헤드 제거 → 1280D 특징맵 출력
                global_pool="avg",
            )
            in_features = self.backbone.num_features  # 1280
        except ImportError:
            raise ImportError("pip install timm 을 먼저 실행하세요.")

        self.embedding_head = EmbeddingHead(
            in_features=in_features,
            embed_dim=embed_dim,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)        # (B, 1280)
        embeddings = self.embedding_head(features)  # (B, 512), L2 정규화됨
        return embeddings

    def get_embedding(self, image_bgr: np.ndarray, device: str = "cpu") -> np.ndarray:
        """
        단일 BGR 이미지 → 512D 임베딩 벡터 (추론용)

        Args:
            image_bgr: OpenCV BGR 이미지
            device:    'cpu' | 'cuda'

        Returns:
            (512,) numpy 배열
        """
        self.eval()
        transform = get_transform(train=False)

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        tensor = transform(image_rgb).unsqueeze(0).to(device)

        with torch.no_grad():
            embedding = self(tensor)

        return embedding.squeeze(0).cpu().numpy()

    def save(self, path: str):
        torch.save(self.state_dict(), path)
        print(f"[모델 저장] {path}")

    def load(self, path: str, device: str = "cpu"):
        self.load_state_dict(torch.load(path, map_location=device))
        print(f"[모델 로드] {path}")
        return self
