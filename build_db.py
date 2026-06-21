"""
FAISS DB 구축 스크립트
학습된 EfficientNet 모델로 val 이미지 임베딩 추출 후 FAISS 인덱스 저장
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import faiss
import json
import argparse
from pathlib import Path
from tqdm import tqdm
import timm
from torchvision import transforms


# ──────────────────────────────────────────────
# 모델 정의 (train.py와 동일해야 함)
# ──────────────────────────────────────────────
class EmbeddingHead(nn.Module):
    def __init__(self, in_features=1280, embed_dim=512, dropout=0.3):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, embed_dim),
    )
    def forward(self, x):
      return F.normalize(self.head(x), p=2, dim=1)  # ← L2 정규화 추가


class BiometricModel(nn.Module):
    def __init__(self, embed_dim=512):
        super().__init__()
        self.backbone = timm.create_model(
            "efficientnet_b0",
            pretrained=False,
            num_classes=0,
            global_pool="avg",
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


def build_db(model_path, data_root, out_index, out_meta, device, batch_size=64):
    print("=" * 50)
    print("FAISS DB 구축 시작")
    print(f"  모델:   {model_path}")
    print(f"  데이터: {data_root}")
    print(f"  디바이스: {device}")
    print("=" * 50)

    # 모델 로드
    model = BiometricModel(embed_dim=512)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    print("모델 로드 완료")

    transform = get_transform()
    data_path = Path(data_root)

    # 이미지 수집
    samples = []
    for dog_dir in sorted(data_path.iterdir()):
        if not dog_dir.is_dir():
            continue
        dog_id = dog_dir.name
        for img_path in list(dog_dir.glob("*.jpg")) + list(dog_dir.glob("*.png")):
            samples.append((str(img_path), dog_id))

    print(f"총 {len(samples)}장 이미지 발견")

    # 배치 임베딩 추출
    all_embeddings = []
    all_metadata = []
    failed = 0

    for i in tqdm(range(0, len(samples), batch_size), desc="임베딩 추출"):
        batch_samples = samples[i:i + batch_size]
        batch_tensors = []
        batch_valid = []

        for img_path, dog_id in batch_samples:
            img = cv2.imread(img_path)
            if img is None:
                failed += 1
                continue
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            tensor = transform(img_rgb)
            batch_tensors.append(tensor)
            batch_valid.append((img_path, dog_id))

        if not batch_tensors:
            continue

        batch = torch.stack(batch_tensors).to(device)
        with torch.no_grad():
            embeddings = model(batch)

        embs = embeddings.cpu().numpy()
        faiss.normalize_L2(embs)

        for j, (img_path, dog_id) in enumerate(batch_valid):
            all_embeddings.append(embs[j])
            all_metadata.append({
                "dog_id":   dog_id,
                "img_path": img_path,
            })

    print(f"임베딩 추출 완료: {len(all_embeddings)}장 (실패: {failed}장)")

    # FAISS 인덱스 생성 및 저장
    embed_dim = 512
    index = faiss.IndexFlatIP(embed_dim)

    all_embeddings_np = np.array(all_embeddings, dtype=np.float32)
    index.add(all_embeddings_np)

    Path(out_index).parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, out_index)
    print(f"FAISS 인덱스 저장: {out_index}")
    print(f"  총 벡터 수: {index.ntotal}")

    with open(out_meta, "w", encoding="utf-8") as f:
        json.dump(all_metadata, f, ensure_ascii=False, indent=2)
    print(f"메타데이터 저장: {out_meta}")

    return index.ntotal


def search_test(index_path, meta_path, query_path, model_path, device):
    """검색 테스트 - 쿼리 이미지로 가장 유사한 강아지 찾기"""
    print("\n검색 테스트")

    # 모델 로드
    model = BiometricModel(embed_dim=512)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    transform = get_transform()

    # 쿼리 임베딩
    img = cv2.imread(query_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    tensor = transform(img_rgb).unsqueeze(0).to(device)
    with torch.no_grad():
        query_emb = model(tensor).cpu().numpy().astype(np.float32)
    faiss.normalize_L2(query_emb)

    # FAISS 검색
    index = faiss.read_index(index_path)
    with open(meta_path, encoding="utf-8") as f:
        metadata = json.load(f)

    similarities, indices = index.search(query_emb, 5)

    print(f"쿼리: {query_path}")
    print("Top 5 결과:")
    for rank, (sim, idx) in enumerate(zip(similarities[0], indices[0]), 1):
        meta = metadata[idx]
        print(f"  {rank}. dog_id={meta['dog_id']}  유사도={sim:.4f}")


def main():
    parser = argparse.ArgumentParser(description="FAISS DB 구축")
    parser.add_argument("--model",      default="D:/pawId/models/semi_hard.pt")
    parser.add_argument("--data_root",  default="D:/pawId/processed/train")
    parser.add_argument("--out_index",  default="D:/pawId/db/faiss.index")
    parser.add_argument("--out_meta",   default="D:/pawId/db/metadata.json")
    parser.add_argument("--device",     default="cuda")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--test_query", default=None, help="검색 테스트용 이미지 경로")
    args = parser.parse_args()

    total = build_db(
        model_path  = args.model,
        data_root   = args.data_root,
        out_index   = args.out_index,
        out_meta    = args.out_meta,
        device      = args.device,
        batch_size  = args.batch_size,
    )

    print(f"\n완료! DB에 {total}개 벡터 저장됨")
    print("다음 단계: python eval.py")

    if args.test_query:
        search_test(
            index_path  = args.out_index,
            meta_path   = args.out_meta,
            query_path  = args.test_query,
            model_path  = args.model,
            device      = args.device,
        )


if __name__ == "__main__":
    main()
