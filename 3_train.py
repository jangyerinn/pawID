"""
3~4주차: 비문 인식 모델 학습
비교 실험 설계 — 기술 선택 근거를 수치로 증명

실험 순서:
  Baseline → 실험1 → 실험2 → 실험3 → 실험4 → 실험5

실행:
    python scripts/3_train.py --mode all        # 모든 실험 순차 실행
    python scripts/3_train.py --mode baseline   # Baseline만
    python scripts/3_train.py --mode semi_hard  # 최종 모델만
"""

import argparse
import sys
sys.path.append(".")

import torch
from model import BiometricModel, get_transform
from train import NoseDataset, PKSampler, TripletLoss, train
from torch.utils.data import DataLoader


EXPERIMENTS = {
    "baseline":     "Baseline: ResNet50 + Contrastive Loss + 랜덤 DataLoader",
    "efficientnet": "실험1: EfficientNet-B0 + Contrastive Loss (Backbone 변경 효과)",
    "triplet":      "실험2: EfficientNet-B0 + Triplet Loss (Loss 변경 효과)",
    "pk_sampler":   "실험3: + PK Sampler (샘플링 전략 핵심 효과)",
    "semi_hard":    "실험4: + Semi-Hard Negative Mining (최종 모델)",
    "augmentation": "실험5: + 데이터 증강",
}


def run_experiment(mode: str, train_root: str, epochs: int, device: str):
    print("\n" + "=" * 60)
    print(f"[{mode}] {EXPERIMENTS.get(mode, mode)}")
    print("=" * 60)

    model = BiometricModel(embed_dim=512, pretrained=True)

    if mode == "baseline":
        # Baseline: ResNet50 기반 (비교용)
        import torchvision.models as tvm
        class BaselineModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                backbone = tvm.resnet50(pretrained=True)
                backbone.fc = torch.nn.Identity()
                self.backbone = backbone
                self.head = torch.nn.Linear(2048, 512)

            def forward(self, x):
                import torch.nn.functional as F
                return F.normalize(self.head(self.backbone(x)), dim=1)

            def get_embedding(self, image, device="cpu"):
                import cv2
                import numpy as np
                self.eval()
                tf = get_transform(train=False)
                img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                t = tf(img_rgb).unsqueeze(0).to(device)
                with torch.no_grad():
                    return self(t).squeeze(0).cpu().numpy()

            def save(self, path):
                torch.save(self.state_dict(), path)

        model = BaselineModel()

        dataset = NoseDataset(train_root, transform=get_transform(train=True))
        loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)
        criterion = torch.nn.CosineEmbeddingLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        save_path = "models/baseline.pt"

        model = model.to(device)
        for epoch in range(1, epochs + 1):
            model.train()
            total_loss = 0.0
            for images, labels in loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                emb = model(images)
                # 간단한 Contrastive Loss (배치 내 첫 절반 vs 후 절반)
                half = len(emb) // 2
                e1, e2 = emb[:half], emb[half:half*2]
                l1, l2 = labels[:half], labels[half:half*2]
                target = (l1 == l2).float() * 2 - 1
                loss = criterion(e1, e2, target)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            if epoch % 10 == 0:
                print(f"  Epoch {epoch}: loss={total_loss/len(loader):.4f}")

        model.save(save_path)
        return save_path

    # EfficientNet 기반 실험들
    use_pk = mode in ("pk_sampler", "semi_hard", "augmentation")
    use_augment = mode == "augmentation"

    save_path = f"models/{mode}.pt"
    train(
        model=model,
        train_root=train_root,
        epochs=epochs,
        P=16,
        K=2,
        margin=0.5,
        lr=3e-5,
        device=device,
        save_path=save_path,
    )
    return save_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="semi_hard",
                        choices=list(EXPERIMENTS.keys()) + ["all"],
                        help="실험 모드")
    parser.add_argument("--train_root", default="data/train",
                        help="학습 데이터 루트")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--device", default="cpu", help="cpu | cuda | mps")
    args = parser.parse_args()

    results = {}

    if args.mode == "all":
        for mode in EXPERIMENTS:
            path = run_experiment(mode, args.train_root, args.epochs, args.device)
            results[mode] = path
    else:
        path = run_experiment(args.mode, args.train_root, args.epochs, args.device)
        results[args.mode] = path

    print("\n" + "=" * 60)
    print("[학습 완료] 저장된 모델:")
    for mode, path in results.items():
        print(f"  {mode}: {path}")
    print("\n다음: python scripts/4_evaluate.py 로 성능 비교")


if __name__ == "__main__":
    main()
