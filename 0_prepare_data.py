import argparse, shutil, csv, cv2, numpy as np
from pathlib import Path
from collections import defaultdict
try:
    from tqdm import tqdm
except:
    def tqdm(x, **k): return x

BASE_DIR      = Path("D:/pawId/pet_biometric_challenge_2022")
OUT_DIR       = Path("D:/pawId/processed")
TRAIN_IMG_DIR = BASE_DIR / "train" / "images"
TRAIN_CSV     = BASE_DIR / "train" / "train_data.csv"
VAL_IMG_DIR   = BASE_DIR / "validation" / "images"
VAL_CSV       = BASE_DIR / "validation" / "valid_data.csv"
MIN_SHARPNESS  = 100.0
MIN_BRIGHTNESS = 40.0

def load_train_csv(path):
    d = defaultdict(list)
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.reader(f)
        next(r)
        for row in r:
            if len(row) >= 2:
                d[row[0].strip()].append(row[1].strip().replace("*", "_"))
    return dict(d)

def load_val_csv(path):
    """val CSV: imageA, imageB 쌍 형식 → 고유 파일명 집합 반환"""
    pairs = []
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.reader(f)
        next(r)
        for row in r:
            if len(row) >= 2:
                a = row[0].strip().replace("*", "_")
                b = row[1].strip().replace("*", "_")
                pairs.append((a, b))
    return pairs

def process(img_dir, csv_path, out_dir, name, use_iqa=False, check_only=False):
    print("\n" + "="*50)
    print("[" + name.upper() + "]")
    print("  img : " + str(img_dir))
    print("  csv : " + str(csv_path))
    if not csv_path.exists():
        print("  ERROR: CSV not found!")
        return
    if not img_dir.exists():
        print("  ERROR: images folder not found!")
        return
    d = load_train_csv(csv_path)
    total = sum(len(v) for v in d.values())
    counts = [len(v) for v in d.values()]
    missing = [f for fs in d.values() for f in fs if not (img_dir / f).exists()]
    print("  dogs=" + str(len(d)) + "  imgs=" + str(total) + "  missing=" + str(len(missing)))
    print("  per dog: min=" + str(min(counts)) + " max=" + str(max(counts)) + " avg=" + str(round(np.mean(counts),1)))
    if missing:
        print("  first 3 missing:")
        for f in missing[:3]:
            print("    - " + f)
    if check_only:
        return
    copied = skip_miss = skip_iqa = 0
    for dog_id, files in tqdm(d.items(), desc=name):
        (out_dir / dog_id).mkdir(parents=True, exist_ok=True)
        for f in files:
            src = img_dir / f
            if not src.exists():
                skip_miss += 1
                continue
            if use_iqa:
                img = cv2.imread(str(src))
                if img is not None:
                    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    s = cv2.Laplacian(g, cv2.CV_64F).var()
                    b = g.mean()
                    if s < MIN_SHARPNESS or b < MIN_BRIGHTNESS:
                        skip_iqa += 1
                        continue
            shutil.copy2(src, out_dir / dog_id / f)
            copied += 1
    print("  copied=" + str(copied) + "  skip_miss=" + str(skip_miss) + "  skip_iqa=" + str(skip_iqa))

def process_val(img_dir, csv_path, out_dir, use_iqa=False, check_only=False):
    """val: (imageA, imageB) 쌍 CSV → 고유 이미지를 flat하게 복사 + pairs.csv 저장"""
    print("\n" + "="*50)
    print("[VAL]")
    print("  img : " + str(img_dir))
    print("  csv : " + str(csv_path))
    if not csv_path.exists():
        print("  ERROR: CSV not found!")
        return
    if not img_dir.exists():
        print("  ERROR: images folder not found!")
        return
    pairs = load_val_csv(csv_path)
    unique_files = sorted({f for pair in pairs for f in pair})
    missing = [f for f in unique_files if not (img_dir / f).exists()]
    print("  pairs=" + str(len(pairs)) + "  unique_imgs=" + str(len(unique_files)) + "  missing=" + str(len(missing)))
    if missing:
        print("  first 3 missing:")
        for f in missing[:3]:
            print("    - " + f)
    if check_only:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    copied = skip_miss = skip_iqa = 0
    for f in tqdm(unique_files, desc="val"):
        src = img_dir / f
        if not src.exists():
            skip_miss += 1
            continue
        if use_iqa:
            img = cv2.imread(str(src))
            if img is not None:
                g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                s = cv2.Laplacian(g, cv2.CV_64F).var()
                b = g.mean()
                if s < MIN_SHARPNESS or b < MIN_BRIGHTNESS:
                    skip_iqa += 1
                    continue
        shutil.copy2(src, out_dir / f)
        copied += 1
    # pairs.csv 저장 (학습/평가 스크립트에서 활용)
    pairs_out = out_dir / "pairs.csv"
    with open(pairs_out, "w", newline="", encoding="utf-8") as pf:
        w = csv.writer(pf)
        w.writerow(["imageA", "imageB"])
        w.writerows(pairs)
    print("  copied=" + str(copied) + "  skip_miss=" + str(skip_miss) + "  skip_iqa=" + str(skip_iqa))
    print("  pairs.csv saved -> " + str(pairs_out))

def main():
    global MIN_SHARPNESS, MIN_BRIGHTNESS
    p = argparse.ArgumentParser()
    p.add_argument("--iqa",        action="store_true")
    p.add_argument("--check_only", action="store_true")
    p.add_argument("--split", default="both", choices=["train", "val", "both"])
    args = p.parse_args()
    if args.split in ("train", "both"):
        process(TRAIN_IMG_DIR, TRAIN_CSV, OUT_DIR / "train", "train", args.iqa, args.check_only)
    if args.split in ("val", "both"):
        process_val(VAL_IMG_DIR, VAL_CSV, OUT_DIR / "val", args.iqa, args.check_only)
    if not args.check_only:
        print("\nDone! Next: python 2_finetune.py")

if __name__ == "__main__": 
    main()