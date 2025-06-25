import os
import cv2
import torch
import argparse
import numpy as np
from craft_text_detector import Craft
from pathlib import Path
from PIL import Image
import unicodedata  # ✅ 한글 깨짐 방지용

# ------------------------
# 파일명 정규화 함수
# ------------------------
def sanitize_filename(text):
    return unicodedata.normalize("NFC", text)

# ------------------------
# 박스 분할 함수 (좌우 등분)
# ------------------------
def split_box(box, num_chars):
    x_min = int(min(pt[0] for pt in box))
    x_max = int(max(pt[0] for pt in box))
    y_min = int(min(pt[1] for pt in box))
    y_max = int(max(pt[1] for pt in box))
    
    width = x_max - x_min
    step = width // num_chars

    new_boxes = []
    for i in range(num_chars):
        new_x_min = x_min + i * step
        new_x_max = x_min + (i + 1) * step
        new_box = [
            [new_x_min, y_min], [new_x_max, y_min],
            [new_x_max, y_max], [new_x_min, y_max]
        ]
        new_boxes.append(new_box)
    return new_boxes

# ------------------------
# 인자 파서: --mode train / val
# ------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, choices=['train', 'val'], required=True)
args = parser.parse_args()

# ------------------------
# 경로 설정
# ------------------------
script_path = Path(__file__).resolve()
project_root = script_path.parent.parent

image_dir = project_root / f"data/eng_images/{args.mode}"
label_dir = project_root / f"data/eng_labels/{args.mode}"
char_image_dir = project_root / f"data/char_images/{args.mode}"
char_label_dir = project_root / f"data/char_labels/{args.mode}"
char_image_dir.mkdir(parents=True, exist_ok=True)
char_label_dir.mkdir(parents=True, exist_ok=True)

# ------------------------
# CRAFT 초기화
# ------------------------
craft = Craft(output_dir='craft_output', crop_type="box", cuda=torch.cuda.is_available())

# ------------------------
# 문자 분리 실행
# ------------------------
for fname in os.listdir(image_dir):
    if not fname.lower().endswith(('.jpg', '.png')):
        continue

    # ✅ 파일명 깨짐 복구 (안 깨졌으면 그대로 사용)
    try:
        fname_fixed = fname.encode('cp437').decode('utf-8')
    except (UnicodeEncodeError, UnicodeDecodeError):
        fname_fixed = fname  # 그대로 사용

    fname = fname_fixed  # 교체

    image_stem = os.path.splitext(fname)[0]
    base_name = image_stem if "_" in image_stem else image_stem + "_1"
    filename_prefix = sanitize_filename(base_name)  # ✅ 여기 적용

    label_path = label_dir / f"{image_stem}.txt"
    if not label_path.exists():
        print(f"❌ 라벨 없음: {label_path.name}")
        continue

    with open(label_path, "r", encoding="utf-8") as f:
        label_text = f.read().strip()

    image_path = image_dir / fname
    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)

    # CRAFT 실행
    prediction_result = craft.detect_text(image_np)
    boxes = prediction_result["boxes"]
    boxes = sorted(boxes, key=lambda b: b[0][0])  # 좌→우 정렬

    # 자동 박스 분할 처리
    if len(boxes) == 1 and len(label_text) > 1:
        print(f"⚠️ 1개 박스 자동 분할 → {len(label_text)} 문자: {fname}")
        boxes = split_box(boxes[0], len(label_text))
    elif len(boxes) == 2 and len(label_text) > 2:
        print(f"⚠️ 2개 박스 자동 분할 → 총 {len(label_text)} 문자: {fname}")
        new_boxes = []
        n1 = len(label_text) // 2
        n2 = len(label_text) - n1
        for box, n in zip(boxes, [n1, n2]):
            new_boxes.extend(split_box(box, n))
        boxes = new_boxes

    if len(boxes) != len(label_text):
        print(f"❌ {fname} - 라벨 {len(label_text)}개 vs 박스 {len(boxes)}개")
        continue

    for i, (box, char) in enumerate(zip(boxes, label_text)):
        x_min = int(min([pt[0] for pt in box]))
        x_max = int(max([pt[0] for pt in box]))
        y_min = int(min([pt[1] for pt in box]))
        y_max = int(max([pt[1] for pt in box]))
        cropped = image_np[y_min:y_max, x_min:x_max]

        # ✅ 깨짐 방지된 이름으로 저장
        char_img_path = char_image_dir / f"{filename_prefix}_{i}.jpg"
        char_lbl_path = char_label_dir / f"{filename_prefix}_{i}.txt"

        cv2.imwrite(str(char_img_path), cropped)
        with open(char_lbl_path, "w", encoding="utf-8") as f:
            f.write(char)

    print(f"✅ {fname} → {len(boxes)} 문자 분리 완료")
