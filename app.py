"""
PawID Gradio 웹앱 - 최종 버전 (UI 개선판)
- 모델: semi_hard_texture.pt (CLAHE + 160px, 150 에폭)
- 다중 사진 등록 + Top-5 후보 + 견종/색상/특징 + Similarity-CAM(타원 마스크)
- 커스텀 테마 + CSS로 UI 정리
"""

import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, cv2, faiss, json, argparse
from pathlib import Path
from torchvision import transforms
import timm, gradio as gr
from PIL import Image

BASE_DIR   = Path(__file__).resolve().parent
MODEL_PATH = str(BASE_DIR / "models" / "semi_hard_texture.pt")
YOLO_PATH  = str(BASE_DIR / "models" / "yolo_nose_detector.pt")
DB_INDEX   = str(BASE_DIR / "db" / "faiss_texture.index")
DB_META    = str(BASE_DIR / "db" / "metadata_texture.json")
DB_IMG_DIR = str(BASE_DIR / "db" / "images_texture")
IMG_SIZE   = 160
THRESHOLD  = 0.45
TOPK       = 5

NAVY  = "#1B2A4A"
TERRA = "#E8743B"

CUSTOM_CSS = """
.gradio-container { font-family: 'Inter', 'Apple SD Gothic Neo', sans-serif !important; max-width: 1280px !important; margin: auto; }
#pawid-header {
    background: linear-gradient(135deg, #1B2A4A 0%, #2A3F6B 100%);
    border-radius: 16px; padding: 28px 32px; margin-bottom: 18px;
    color: white;
}
#pawid-header h1 { margin: 0 0 6px 0; font-size: 30px; font-weight: 800; color: white;}
#pawid-header p { margin: 0; opacity: 0.85; font-size: 15px; color: white;}
.pawid-card { border-radius: 14px !important; }
.pawid-result { border-radius: 14px !important; padding: 4px !important; }
footer { display: none !important; }
.tabitem { padding-top: 8px !important; }
"""

THEME = gr.themes.Soft(
    primary_hue=gr.themes.colors.orange,
    secondary_hue=gr.themes.colors.blue,
    neutral_hue=gr.themes.colors.slate,
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "sans-serif"],
).set(
    button_primary_background_fill=TERRA,
    button_primary_background_fill_hover="#C85A24",
    button_primary_text_color="white",
    block_title_text_weight="600",
    block_border_width="1px",
    block_shadow="0 1px 4px rgba(0,0,0,0.06)",
)


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


def get_transform(img_size=IMG_SIZE):
    return transforms.Compose([
        transforms.ToPILImage(), transforms.Resize((img_size, img_size)), transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


class PawIDSystem:
    def __init__(self, model_path, yolo_path, db_index, db_meta, threshold, device="cpu"):
        self.threshold = threshold
        self.device    = device
        self.transform = get_transform()

        self.model = BiometricModel(embed_dim=512)
        self.model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
        self.model = self.model.to(device); self.model.eval()

        self.yolo = None
        try:
            from ultralytics import YOLO
            if Path(yolo_path).exists():
                self.yolo = YOLO(yolo_path)
        except Exception as e:
            print("YOLOv8 로드 실패:", e)

        if Path(db_index).exists() and Path(db_meta).exists():
            self.index = faiss.read_index(db_index)
            with open(db_meta, encoding="utf-8") as f:
                self.metadata = json.load(f)
        else:
            self.index    = faiss.IndexFlatIP(512)
            self.metadata = []

        Path(DB_IMG_DIR).mkdir(parents=True, exist_ok=True)
        print("PawID 준비 완료 | DB:", self.index.ntotal, "마리 | YOLOv8:", "ON" if self.yolo else "OFF")

    def iqa_check(self, img_bgr):
        gray   = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        sharp  = cv2.Laplacian(gray, cv2.CV_64F).var()
        bright = gray.mean()
        if sharp < 15:
            return False, "사진이 흔들렸습니다. (선명도: " + str(round(sharp)) + ")"
        if bright < 10:
            return False, "너무 어둡습니다. (밝기: " + str(round(bright)) + ")"
        return True, "품질 양호"

    def detect_and_crop(self, img_bgr):
        if self.yolo is None: return img_bgr, None
        try:
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            results = self.yolo.predict(img_rgb, conf=0.25, verbose=False)
            boxes = results[0].boxes
            if len(boxes) == 0: return img_bgr, None
            best = boxes.conf.argmax()
            x1,y1,x2,y2 = map(int, boxes.xyxy[best].cpu().numpy())
            h,w = img_bgr.shape[:2]; pad = 15
            x1,y1 = max(0,x1-pad), max(0,y1-pad)
            x2,y2 = min(w,x2+pad), min(h,y2+pad)
            return img_bgr[y1:y2, x1:x2], (x1,y1,x2,y2)
        except Exception as e:
            print("YOLO 에러:", e); return img_bgr, None

    def get_embedding(self, img_bgr):
        img_rgb = apply_clahe(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        tensor  = self.transform(img_rgb).unsqueeze(0).to(self.device)
        with torch.no_grad():
            emb = self.model(tensor).cpu().numpy().astype(np.float32)
        faiss.normalize_L2(emb)
        return emb

    def register_multi(self, img_pils, dog_name, owner_name, owner_phone, breed, color, features):
        imgs = [p for p in (img_pils or []) if p is not None]
        if not imgs:
            return "⚠️ 사진을 1장 이상 업로드해주세요. (여러 장 등록 시 인식률 향상)"
        if not dog_name or not owner_name or not owner_phone:
            return "⚠️ 강아지 이름, 견주 이름, 전화번호를 모두 입력해주세요."

        embs, cropped_list, yolo_hits = [], [], 0
        for img_pil in imgs:
            img_bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
            cropped, bbox = self.detect_and_crop(img_bgr)
            if bbox: yolo_hits += 1
            ok, msg = self.iqa_check(cropped)
            if not ok: continue
            embs.append(self.get_embedding(cropped))
            cropped_list.append(cropped)

        if not embs:
            return "❌ 모든 사진의 품질이 낮아 등록할 수 없습니다. 더 선명한 사진으로 시도해주세요."

        avg_emb = np.mean(np.vstack(embs), axis=0).reshape(1, -1).astype(np.float32)
        faiss.normalize_L2(avg_emb)

        new_id = self.index.ntotal
        self.index.add(avg_emb)
        img_save_path = str(Path(DB_IMG_DIR) / (str(new_id) + ".jpg"))
        cv2.imwrite(img_save_path, cropped_list[0])

        self.metadata.append({
            "dog_id": str(new_id), "dog_name": dog_name, "owner": owner_name,
            "phone": owner_phone, "breed": breed or "미입력",
            "color": color or "미입력", "features": features or "특이사항 없음",
            "img_path": img_save_path, "n_photos": len(embs),
        })
        Path(DB_INDEX).parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, DB_INDEX)
        with open(DB_META, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)

        return (
            "### ✅ 등록 완료\n\n"
            "| 항목 | 내용 |\n|---|---|\n"
            "| 강아지 | **" + dog_name + "** |\n"
            "| 견주 | " + owner_name + " |\n"
            "| 사용된 사진 | " + str(len(embs)) + "장 (평균 임베딩) |\n"
            "| YOLO 코 탐지 | " + str(yolo_hits) + "/" + str(len(imgs)) + " |\n"
            "| 현재 총 등록 | **" + str(self.index.ntotal) + "마리** |"
        )

    def identify(self, img_pil):
        empty = (None, None, None)
        if img_pil is None: return ("이미지를 업로드해주세요.",) + empty
        img_bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        cropped, bbox = self.detect_and_crop(img_bgr)
        yolo_msg = "✅ YOLOv8 코 탐지 성공" if bbox else "⚠️ YOLOv8 탐지 실패 - 원본 사용"

        ok, iqa_msg = self.iqa_check(cropped)
        if not ok: return ("### ⚠️ 품질 문제\n" + iqa_msg,) + empty
        if self.index.ntotal == 0: return ("### ℹ️ DB에 등록된 강아지가 없습니다.",) + empty

        emb = self.get_embedding(cropped)
        k = min(TOPK, self.index.ntotal)
        sims, idxs = self.index.search(emb, k)
        top1_sim = float(sims[0][0])

        if top1_sim >= self.threshold:
            md = "### 🔍 가장 유사한 후보를 찾았습니다 (고확신)\n\n"
        else:
            md = "### ❔ 확신할 수 있는 후보가 없습니다 — 아래 후보를 직접 비교해주세요\n\n"

        md += "| 순위 | 이름 | 유사도 | 견종 | 색상 | 특징 | 견주 / 연락처 |\n"
        md += "|---|---|---|---|---|---|---|\n"
        for i, (s, idx) in enumerate(zip(sims[0], idxs[0])):
            m = self.metadata[int(idx)]
            mark = " 🏆" if i == 0 else ""
            md += ("| " + str(i+1) + mark + " | **" + m.get("dog_name","?") + "** | " +
                   str(round(float(s),3)) + " | " + m.get("breed","미입력") + " | " +
                   m.get("color","미입력") + " | " + m.get("features","-") + " | " +
                   m.get("owner","?") + " / " + m.get("phone","?") + " |\n")

        md += ("\n> ⚠️ **자동 확정이 아닙니다.** 코 유사도뿐 아니라 견종·색상·특징이 실제 발견한 강아지와 "
               "일치하는지 함께 확인한 뒤 최종 판단해주세요.\n\n*" + yolo_msg + "*")

        if bbox:
            x1,y1,x2,y2 = bbox
            vis = img_bgr.copy()
            cv2.rectangle(vis,(x1,y1),(x2,y2),(46,116,232),3)
            vis_pil = Image.fromarray(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
        else:
            vis_pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))

        gallery_items = []
        for i, idx in enumerate(idxs[0]):
            m = self.metadata[int(idx)]
            db_img = self._load_db_image(m)
            if db_img is not None:
                db_pil = Image.fromarray(cv2.cvtColor(db_img, cv2.COLOR_BGR2RGB))
                caption = (str(i+1) + "위 " + m.get("dog_name","?") +
                          " (" + str(round(float(sims[0][i]),3)) + ") " +
                          m.get("breed","?") + "/" + m.get("color","?"))
                gallery_items.append((db_pil, caption))

        target_emb = self.index.reconstruct(int(idxs[0][0]))
        heatmap = self.generate_heatmap(cropped, target_emb)

        return md, vis_pil, gallery_items, heatmap

    def _load_db_image(self, meta):
        path = meta.get("img_path")
        if not path or not Path(path).exists(): return None
        return cv2.imread(path)

    def generate_heatmap(self, img_bgr, target_emb_np):
        try:
            img_rgb = apply_clahe(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
            tensor  = self.transform(img_rgb).unsqueeze(0).to(self.device)

            grads_list, acts_list = [], []
            target_layer = list(self.model.backbone.blocks.children())[5]
            h1 = target_layer.register_forward_hook(lambda m,i,o: acts_list.append(o))
            h2 = target_layer.register_full_backward_hook(lambda m,gi,go: grads_list.append(go[0]))

            q_emb  = self.model(tensor)
            target = torch.tensor(target_emb_np, dtype=torch.float32, device=self.device).view(1,-1)
            target = target / (target.norm(p=2, dim=1, keepdim=True) + 1e-8)
            score  = (q_emb * target).sum()
            self.model.zero_grad(); score.backward()
            h1.remove(); h2.remove()
            if not grads_list: return None

            weights = grads_list[0].mean(dim=[2,3], keepdim=True)
            cam = F.relu((weights * acts_list[0]).sum(dim=1)).squeeze().detach().cpu().numpy()
            cam = cam - cam.min()
            if cam.max() < 1e-8: return None
            cam = cam / cam.max()
            h, w = img_bgr.shape[:2]
            cam_r = cv2.resize(cam, (w, h))
            mh, mw = cam_r.shape
            mask = np.zeros((mh, mw), dtype=np.uint8)
            center = (mw // 2, mh // 2)
            axes = (int(mw * 0.36), int(mh * 0.42))
            cv2.ellipse(mask, center, axes, 0, 0, 360, 1, -1)
            cam_r = cam_r * mask.astype(np.float32)
            colored = cv2.applyColorMap((cam_r*255).astype(np.uint8), cv2.COLORMAP_JET)
            overlay = cv2.addWeighted(img_bgr, 0.5, colored, 0.5, 0)
            return Image.fromarray(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        except Exception as e:
            print("CAM 에러:", e); return None


def build_app(system):
    with gr.Blocks(title="PawID", theme=THEME, css=CUSTOM_CSS) as app:

        gr.HTML(
            '<div id="pawid-header">'
            '<h1>🐾 PawID</h1>'
            '<p>대조 학습 기반 유기견 비문(코 무늬) 생체 식별 시스템 · '
            '코 사진 한 장으로 견주를 찾아드립니다</p>'
            '</div>'
        )

        with gr.Tabs():
            with gr.Tab("🔍 유기견 식별"):
                with gr.Row():
                    with gr.Column(scale=1):
                        with gr.Group():
                            id_img = gr.Image(label="유기견 사진", type="pil",
                                               sources=["upload","webcam"], elem_classes="pawid-card")
                            id_btn = gr.Button("🔍 식별하기", variant="primary", size="lg")
                        id_result = gr.Markdown(elem_classes="pawid-result")
                    with gr.Column(scale=1):
                        with gr.Group():
                            id_bbox = gr.Image(label="YOLO 탐지 결과", type="pil", height=220)
                        with gr.Group():
                            id_gallery = gr.Gallery(label="Top-5 DB 후보", columns=5, height=170)
                        with gr.Group():
                            id_heat = gr.Image(label="Similarity-CAM (판별 근거)", type="pil", height=220)
                id_btn.click(
                    fn=lambda img: system.identify(img) if system else ("데모 모드", None, [], None),
                    inputs=[id_img], outputs=[id_result, id_bbox, id_gallery, id_heat],
                )

            with gr.Tab("📝 강아지 등록"):
                gr.Markdown(
                    "💡 **여러 장(2장 이상) 등록 시 인식률이 향상됩니다** "
                    "— 검증됨: Rank-1 86.0%→90.5%, Rank-5 95.5%→98.5%"
                )
                with gr.Row():
                    with gr.Column(scale=1):
                        with gr.Group():
                            reg_upload = gr.File(label="강아지 코 사진 업로드 (여러 장 선택 가능)",
                                                  file_count="multiple", file_types=["image"])
                            reg_imgs = gr.Gallery(label="미리보기", columns=4, height=170)
                        with gr.Group():
                            with gr.Row():
                                reg_name  = gr.Textbox(label="강아지 이름", placeholder="예: 초코")
                                reg_owner = gr.Textbox(label="견주 이름", placeholder="예: 홍길동")
                            reg_phone = gr.Textbox(label="전화번호", placeholder="예: 010-1234-5678")
                            with gr.Row():
                                reg_breed = gr.Textbox(label="견종 (선택)", placeholder="예: 말티즈")
                                reg_color = gr.Textbox(label="털 색상 (선택)", placeholder="예: 흰색")
                            reg_features = gr.Textbox(label="특징 (선택)", placeholder="예: 왼쪽 귀 끝 검음, 꼬리 짧음")
                        reg_btn = gr.Button("✅ 등록하기", variant="primary", size="lg")
                    with gr.Column(scale=1):
                        reg_result = gr.Markdown(elem_classes="pawid-result")
                        gr.Markdown(
                            "코 유사도만으로는 헷갈리는 경우가 있어, 견종·색상·특징을 함께 등록하면 "
                            "식별 시 사람이 최종 확인하는 데 도움이 됩니다."
                        )

                def load_and_preview(files):
                    if not files: return []
                    return [Image.open(f.name if hasattr(f,"name") else f) for f in files]

                def do_register(files, n, o, p, b, c, ft):
                    if not system: return "데모 모드"
                    imgs = []
                    if files:
                        for f in files:
                            path = f.name if hasattr(f, "name") else f
                            try:
                                imgs.append(Image.open(path).convert("RGB"))
                            except Exception:
                                pass
                    return system.register_multi(imgs, n, o, p, b, c, ft)

                reg_upload.change(fn=load_and_preview, inputs=[reg_upload], outputs=[reg_imgs])
                reg_btn.click(
                    fn=do_register,
                    inputs=[reg_upload, reg_name, reg_owner, reg_phone, reg_breed, reg_color, reg_features],
                    outputs=[reg_result],
                )

            with gr.Tab("ℹ️ 시스템 정보"):
                db_n = system.index.ntotal if system else 0
                gr.Markdown(
                    "## 모델: `semi_hard_texture.pt` (150 에폭, 수렴 확인)\n"
                    "- 전처리: CLAHE 텍스처 강화 + 160×160\n"
                    "- Threshold: **" + str(THRESHOLD) + "**\n\n"
                    "## 최종 성능\n\n"
                    "| 지표 | 값 |\n|---|---|\n"
                    "| YOLOv8n mAP50 | 0.995 |\n"
                    "| TAR (1:1 검증) | 57.8% |\n"
                    "| Rank-1 (단일사진) | 86.0% |\n"
                    "| **Rank-1 (다중사진)** | **90.5%** |\n"
                    "| Rank-5 (다중사진) | 98.5% |\n\n"
                    "## DB 현황\n등록: **" + str(db_n) + "마리**"
                )
    return app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--threshold", type=float, default=THRESHOLD)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    system = None
    if not args.demo:
        try:
            system = PawIDSystem(MODEL_PATH, YOLO_PATH, DB_INDEX, DB_META, args.threshold, args.device)
        except Exception as e:
            print("로드 실패:", e)
    build_app(system).launch(server_port=args.port, share=args.share, show_error=True)


if __name__ == "__main__":
    main()
