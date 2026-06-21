"""
PawID 성능 평가 + Similarity-CAM

평가 방식:
  Train 데이터에서 마리당 1장을 쿼리(query)로, 나머지를 갤러리(gallery)로 사용
  val CSV는 imageA/imageB 쌍 형식이라 Rank-1 평가에 부적합 → train split 활용
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import faiss
import json
import argparse
import random
from pathlib import Path
from tqdm import tqdm
import timm
from torchvision import transforms


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
            nn.BatchNorm1d(embed_dim),  # ← 이 줄 추가
        )
    def forward(self, x):
        return F.normalize(self.head(x), p=2, dim=1)


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


def load_model(model_path, device):
    model = BiometricModel(embed_dim=512)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
    model = model.to(device)
    model.eval()
    return model


def get_embedding(model, img_bgr, transform, device):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    tensor = transform(img_rgb).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model(tensor).cpu().numpy().astype(np.float32)
    faiss.normalize_L2(emb)
    return emb


# --------------------------------------------------
# 1. 성능 평가 (train split 활용)
# --------------------------------------------------
def evaluate(model, train_root, device, threshold=0.8, max_dogs=1000, seed=42):
    """
    Train 데이터에서 마리당 1장 query / 나머지 gallery 로 Rank-1/5 측정
    max_dogs: 평가할 최대 강아지 수 (전체 6000마리는 오래 걸림)
    """
    random.seed(seed)
    transform = get_transform()
    train_path = Path(train_root)

    # 개체별 이미지 수집
    dog_to_imgs = {}
    for dog_dir in sorted(train_path.iterdir()):
        if not dog_dir.is_dir():
            continue
        imgs = list(dog_dir.glob("*.jpg")) + list(dog_dir.glob("*.png"))
        if len(imgs) >= 2:   # 최소 2장 이상인 개체만
            dog_to_imgs[dog_dir.name] = [str(p) for p in imgs]

    # max_dogs 개체 랜덤 샘플
    all_ids = list(dog_to_imgs.keys())
    if len(all_ids) > max_dogs:
        all_ids = random.sample(all_ids, max_dogs)

    print(f"평가 대상: {len(all_ids)}마리 (이미지 2장 이상)")

    # Gallery 임베딩 구축
    print("Gallery 임베딩 추출 중...")
    gallery_embs = []
    gallery_ids  = []

    for dog_id in tqdm(all_ids, desc="Gallery"):
        imgs = dog_to_imgs[dog_id]
        # 첫 번째 이미지는 query 용으로 남겨두고 나머지를 gallery
        for img_path in imgs[1:]:
            img = cv2.imread(img_path)
            if img is None:
                continue
            emb = get_embedding(model, img, transform, device)
            gallery_embs.append(emb[0])
            gallery_ids.append(dog_id)

    gallery_np = np.array(gallery_embs, dtype=np.float32)
    index = faiss.IndexFlatIP(512)
    index.add(gallery_np)
    print(f"Gallery 구축 완료: {index.ntotal}개 벡터")

    # Query 검색
    print("Query 검색 중...")
    rank1 = rank5 = total = 0
    genuine_sims  = []
    impostor_sims = []

    for dog_id in tqdm(all_ids, desc="Query"):
        imgs = dog_to_imgs[dog_id]
        query_img = cv2.imread(imgs[0])
        if query_img is None:
            continue

        query_emb = get_embedding(model, query_img, transform, device)
        k = min(5, index.ntotal)
        sims, idxs = index.search(query_emb, k)

        retrieved = [gallery_ids[i] for i in idxs[0] if i >= 0]
        top1_sim  = float(sims[0][0])

        if retrieved and retrieved[0] == dog_id:
            rank1 += 1
            genuine_sims.append(top1_sim)
        else:
            impostor_sims.append(top1_sim)

        if dog_id in retrieved:
            rank5 += 1

        total += 1

    r1 = rank1 / total
    r5 = rank5 / total

    print("\n" + "="*45)
    print("[성능 평가 결과]")
    print(f"  Rank-1 Accuracy : {r1:.4f}  (목표: 0.85 이상)")
    print(f"  Rank-5 Accuracy : {r5:.4f}  (목표: 0.95 이상)")
    print(f"  평가 강아지 수  : {total}")
    print("="*45)

    # EER
    if genuine_sims and impostor_sims:
        eer, opt_t = compute_eer(genuine_sims, impostor_sims)
        print(f"\n  최적 Threshold {opt_t:.4f} 을 app.py 에 반영하세요")

    return r1, r5


def compute_eer(genuine, impostor):
    all_sims   = np.array(genuine + impostor)
    all_labels = np.array([1]*len(genuine) + [0]*len(impostor))
    thresholds = np.linspace(all_sims.min(), all_sims.max(), 1000)

    far_list, frr_list = [], []
    for t in thresholds:
        far = (all_sims[all_labels == 0] >= t).mean()
        frr = (all_sims[all_labels == 1] <  t).mean()
        far_list.append(far)
        frr_list.append(frr)

    far_arr = np.array(far_list)
    frr_arr = np.array(frr_list)
    idx = np.abs(far_arr - frr_arr).argmin()
    eer = (far_arr[idx] + frr_arr[idx]) / 2
    opt_t = float(thresholds[idx])

    print(f"  EER             : {eer:.4f}  (목표: 0.10 이하)")
    print(f"  최적 Threshold  : {opt_t:.4f}")

    try:
        import matplotlib.pyplot as plt
        Path("results").mkdir(exist_ok=True)
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].plot(thresholds, far_arr, label="FAR", color="red")
        axes[0].plot(thresholds, frr_arr, label="FRR", color="blue")
        axes[0].axvline(opt_t, color="green", linestyle="--",
                        label=f"EER={eer:.3f} @ t={opt_t:.3f}")
        axes[0].set_xlabel("Threshold"); axes[0].set_ylabel("Error Rate")
        axes[0].set_title("FAR-FRR Curve"); axes[0].legend(); axes[0].grid(True)

        tpr = 1 - frr_arr
        axes[1].plot(far_arr, tpr, color="purple", lw=2)
        axes[1].plot([0,1],[0,1], "k--", alpha=0.5)
        axes[1].scatter([eer],[1-eer], color="green", s=100, zorder=5,
                        label=f"EER={eer:.3f}")
        axes[1].set_xlabel("FAR"); axes[1].set_ylabel("TPR (1-FRR)")
        axes[1].set_title("ROC Curve"); axes[1].legend(); axes[1].grid(True)

        plt.tight_layout()
        plt.savefig("results/roc_curve.png", dpi=150)
        plt.close()
        print(f"  ROC Curve 저장  : results/roc_curve.png")
    except ImportError:
        pass

    return eer, opt_t


# --------------------------------------------------
# 2. Similarity-CAM
# --------------------------------------------------
class SimilarityCAM:
    def __init__(self, model, device):
        self.model    = model
        self.device   = device
        self.gradients   = None
        self.activations = None

        target = list(model.backbone.blocks.children())[4]
        target.register_forward_hook(self._save_activation)
        target.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, out):
        self.activations = out.detach()

    def _save_gradient(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def generate_pair(self, query_bgr, db_bgr, transform):
        self.model.eval()

        def to_tensor(img):
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return transform(rgb).unsqueeze(0).to(self.device)

        q_t = to_tensor(query_bgr)
        d_t = to_tensor(db_bgr)
        q_t.requires_grad_(True)

        q_emb = self.model(q_t)
        with torch.no_grad():
            d_emb = self.model(d_t)

        sim = (q_emb * d_emb).sum()
        self.model.zero_grad()
        sim.backward()

        grads = self.gradients
        acts  = self.activations
        weights = grads.mean(dim=[2, 3], keepdim=True)
        cam = (weights * acts).sum(dim=1).squeeze()
        cam = F.relu(cam).cpu().numpy()
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()

        h, w = query_bgr.shape[:2]
        cam_r = cv2.resize(cam, (w, h))
        mask = np.zeros_like(cam_r)
        mh, mw = mask.shape
        y1, y2 = int(mh * 0.1), int(mh * 0.9)
        x1, x2 = int(mw * 0.1), int(mw * 0.9)
        mask[y1:y2, x1:x2] = 1
        cam_r = cam_r * mask
        colored = cv2.applyColorMap((cam_r * 255).astype(np.uint8), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(query_bgr, 0.5, colored, 0.5, 0)
        return overlay, float(sim.item())


def run_cam(model, train_root, index_path, meta_path, device, n_samples=10):
    transform = get_transform()
    index     = faiss.read_index(index_path)
    with open(meta_path, encoding="utf-8") as f:
        metadata = json.load(f)

    cam_module = SimilarityCAM(model, device)
    out_dir    = Path("results/cam")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 샘플 수집 (마리당 1장)
    samples = []
    for dog_dir in sorted(Path(train_root).iterdir()):
        if not dog_dir.is_dir():
            continue
        imgs = list(dog_dir.glob("*.jpg"))
        if imgs:
            samples.append((str(imgs[0]), dog_dir.name))
        if len(samples) >= n_samples:
            break

    print(f"\nSimilarity-CAM 생성 중 ({len(samples)}장)...")

    for i, (img_path, dog_id) in enumerate(samples):
        query_img = cv2.imread(img_path)
        if query_img is None:
            continue

        query_emb = get_embedding(model, query_img, transform, device)
        sims, idxs = index.search(query_emb, 1)
        top1_meta = metadata[idxs[0][0]]
        top1_sim  = float(sims[0][0])
        db_img    = cv2.imread(top1_meta["img_path"])
        if db_img is None:
            continue

        heatmap, sim = cam_module.generate_pair(query_img, db_img, transform)

        q_r  = cv2.resize(query_img, (224, 224))
        db_r = cv2.resize(db_img,    (224, 224))
        h_r  = cv2.resize(heatmap,   (224, 224))
        combined = np.hstack([q_r, db_r, h_r])

        label = "MATCH" if top1_meta["dog_id"] == dog_id else "NO MATCH"
        color = (0, 255, 0) if label == "MATCH" else (0, 0, 255)
        cv2.putText(combined, label + "  sim=" + str(round(sim, 3)),
                    (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.putText(combined, "Query",       (5, 215),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1)
        cv2.putText(combined, "DB Match",    (229, 215),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1)
        cv2.putText(combined, "CAM Heatmap", (453, 215),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1)

        fname = "cam_" + str(i).zfill(2) + "_dog" + dog_id + "_" + label.replace(" ", "") + ".jpg"
        cv2.imwrite(str(out_dir / fname), combined)
        print("  [" + str(i+1) + "/" + str(len(samples)) + "] dog_id=" + dog_id
              + " -> " + top1_meta["dog_id"]
              + "  sim=" + str(round(sim, 3)) + "  " + label)

    print("\nCAM 저장 완료: results/cam/")


# --------------------------------------------------
# 메인
# --------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",      default="D:/pawId/models/semi_hard.pt")
    parser.add_argument("--train_root", default="D:/pawId/processed/train")
    parser.add_argument("--index",      default="D:/pawId/db/faiss.index")
    parser.add_argument("--meta",       default="D:/pawId/db/metadata.json")
    parser.add_argument("--device",     default="cuda")
    parser.add_argument("--max_dogs",   type=int, default=500,
                        help="평가할 강아지 수 (기본 500, 전체는 6000)")
    parser.add_argument("--n_cam",      type=int, default=10)
    parser.add_argument("--cam_only",   action="store_true")
    parser.add_argument("--eval_only",  action="store_true")
    args = parser.parse_args()

    Path("results").mkdir(exist_ok=True)
    print("모델 로드 중...")
    model = load_model(args.model, args.device)

    if not args.cam_only:
        print("\n[1단계] 성능 평가")
        evaluate(model, args.train_root, args.device, max_dogs=args.max_dogs)

    if not args.eval_only:
        print("\n[2단계] Similarity-CAM")
        run_cam(model, args.train_root, args.index, args.meta, args.device, args.n_cam)

    print("\n완료!")
    print("  ROC Curve  : results/roc_curve.png")
    print("  CAM 히트맵 : results/cam/")
    print("  다음 단계  : python app.py --demo")

if __name__ == "__main__":
    main()