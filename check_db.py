"""
DB 등록 내용 확인 스크립트
실행: python check_db.py
"""
import json
from pathlib import Path

META_PATH = "D:/pawId/db/metadata.json"

with open(META_PATH, encoding="utf-8") as f:
    data = json.load(f)

print("=" * 50)
print("총 등록 수:", len(data))
print("=" * 50)

# 스키마 종류 확인 (build_db.py 로 만든 것 vs Gradio 등록)
bulk_count = sum(1 for d in data if "img_path" in d and "dog_name" not in d)
gradio_count = sum(1 for d in data if "dog_name" in d)

print("bulk(학습데이터) 등록:", bulk_count)
print("Gradio 앱 등록:", gradio_count)
print()

print("[처음 5개 항목]")
for d in data[:5]:
    print(d)

print()
print("[마지막 5개 항목] (최근 Gradio 등록)")
for d in data[-5:]:
    print(d)
