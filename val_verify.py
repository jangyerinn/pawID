"""
Val CSV (imageA, imageB) 쌍 검증 테스트
- 모든 쌍이 같은 강아지의 다른 사진
- 유사도 > threshold 이면 올바르게 인식한 것
- 통과율 = TAR (True Acceptance Rate) = 1 - FRR
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import faiss
import csv
import argparse
from pathlib import Path
from tqdm import tqdm
import timm
from torchvision import transforms


# --------------------------------------------------
# 모델 정의 (ArcFace 버전과 동일)
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
            nn.BatchNorm1d(embed_dim),
      
        )
    def forward(self, x):
        return self.head(x)


class BiometricModel(nn.Module):
    def __init__(self, embed_dim=512):
        super().__init__()
        self.backbone = timm.create_model(
            "efficientnet_b0", pretrained=False,
            num_classes=0, global_pool="avg",
        )
        self.embedding_head = EmbeddingHead(
            in_features=self.backbone.num_features,
            embed_dim=embed_dim,
        )
    def forward(self, x):
        return self.embedding_head(self.backbone(x))


def get_transform():
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def get_embedding(model, img_bgr, transform, device):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    tensor  = transform(img_rgb).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model(tensor)
        emb = F.normalize(emb, p=2, dim=1)
    return emb.cpu().numpy().astype(np.float32)


# --------------------------------------------------
# 메인
# --------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Val 쌍 검증 테스트")
    parser.add_argument("--model",    default="D:/pawId/models/arcface.pt")
    parser.add_argument("--val_csv",  default="D:/pawId/pet_biometric_challenge_2022/validation/valid_data.csv")
    parser.add_argument("--val_img",  default="D:/pawId/pet_biometric_challenge_2022/validation/images")
    parser.add_argument("--threshold",type=float, default=0.8967)
    parser.add_argument("--device",   default="cuda")
    parser.add_argument("--save_vis", action="store_true", help="시각화 이미지 저장")
    parser.add_argument("--n_vis",    type=int, default=20, help="저장할 시각화 수")
    args = parser.parse_args()

    # 모델 로드
    model = BiometricModel(embed_dim=512)
    model.load_state_dict(torch.load(args.model, map_location=args.device, weights_only=False))
    model = model.to(args.device)
    model.eval()
    transform = get_transform()
    print("모델 로드 완료")

    # Val CSV 파싱
    val_img_dir = Path(args.val_img)
    pairs = []
    with open(args.val_csv, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)  # 헤더(imageA, imageB) 건너뜀
        for row in reader:
            if len(row) >= 2:
                imgA = row[0].strip().replace("*", "_")
                imgB = row[1].strip().replace("*", "_")
                pairs.append((imgA, imgB))

    print(f"총 쌍 수: {len(pairs)}")

    # 유사도 계산
    similarities = []
    passed = 0
    failed = 0
    load_error = 0
    vis_data = []

    for imgA_name, imgB_name in tqdm(pairs, desc="검증"):
        imgA_path = val_img_dir / imgA_name
        imgB_path = val_img_dir / imgB_name

        imgA = cv2.imread(str(imgA_path))
        imgB = cv2.imread(str(imgB_path))

        if imgA is None or imgB is None:
            load_error += 1
            continue

        embA = get_embedding(model, imgA, transform, args.device)
        embB = get_embedding(model, imgB, transform, args.device)

        sim = float(np.dot(embA[0], embB[0]))
        similarities.append(sim)

        if sim >= args.threshold:
            passed += 1
        else:
            failed += 1

        if args.save_vis and len(vis_data) < args.n_vis:
            vis_data.append((imgA, imgB, sim, sim >= args.threshold))

    # 결과 출력
    total = len(similarities)
    tar   = passed / total if total > 0 else 0
    frr   = failed / total if total > 0 else 0

    print("\n" + "="*50)
    print("[Val 쌍 검증 결과] (모든 쌍 = 같은 강아지)")
    print(f"  총 쌍 수    : {total}")
    print(f"  통과(PASS)  : {passed}  ({tar*100:.1f}%)")
    print(f"  실패(FAIL)  : {failed}  ({frr*100:.1f}%)")
    print(f"  로드 에러   : {load_error}")
    print(f"  평균 유사도 : {np.mean(similarities):.4f}")
    print(f"  최소 유사도 : {np.min(similarities):.4f}")
    print(f"  최대 유사도 : {np.max(similarities):.4f}")
    print(f"  Threshold   : {args.threshold}")
    print(f"  TAR (True Acceptance Rate) : {tar:.4f}")
    print(f"  FRR (False Rejection Rate) : {frr:.4f}")
    print("="*50)

    # 유사도 분포 저장
    try:
        import matplotlib.pyplot as plt
        Path("results").mkdir(exist_ok=True)
        plt.figure(figsize=(8, 5))
        plt.hist(similarities, bins=50, color="steelblue", alpha=0.7, edgecolor="black")
        plt.axvline(args.threshold, color="red", linestyle="--",
                    label=f"Threshold={args.threshold:.3f}")
        plt.xlabel("Cosine Similarity")
        plt.ylabel("Count")
        plt.title(f"Val Genuine Pair Similarity Distribution\nTAR={tar:.3f}  FRR={frr:.3f}")
        plt.legend()
        plt.tight_layout()
        plt.savefig("results/val_similarity_dist.png", dpi=150)
        plt.close()
        print("유사도 분포 저장: results/val_similarity_dist.png")
    except ImportError:
        pass

    # 시각화 저장
    if args.save_vis and vis_data:
        vis_dir = Path("results/val_vis")
        vis_dir.mkdir(parents=True, exist_ok=True)
        for i, (imgA, imgB, sim, ok) in enumerate(vis_data):
            A = cv2.resize(imgA, (224, 224))
            B = cv2.resize(imgB, (224, 224))
            combined = np.hstack([A, B])
            label = "PASS" if ok else "FAIL"
            color = (0, 255, 0) if ok else (0, 0, 255)
            cv2.putText(combined, f"{label}  sim={sim:.3f}",
                        (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.putText(combined, "imageA", (5, 215),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
            cv2.putText(combined, "imageB", (229, 215),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
            fname = f"{i:02d}_{label}_sim{sim:.3f}.jpg"
            cv2.imwrite(str(vis_dir / fname), combined)
        print(f"시각화 저장: results/val_vis/ ({len(vis_data)}장)")


if __name__ == "__main__":
    main()
