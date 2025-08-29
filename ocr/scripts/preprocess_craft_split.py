import os
import cv2
import torch
import argparse
import numpy as np
from craft_text_detector import Craft
from pathlib import Path
from PIL import Image
import unicodedata

# ------------------------
# 파일명 정규화 함수
# ------------------------
def sanitize_filename(text):
    return unicodedata.normalize("NFC", text)

# ------------------------
# 박스 면적 및 너비 계산 함수
# ------------------------
def get_box_area(box):
    return (max(p[0] for p in box) - min(p[0] for p in box)) * \
           (max(p[1] for p in box) - min(p[1] for p in box))

def get_box_width(box):
    return max(p[0] for p in box) - min(p[0] for p in box)

# ------------------------
# 이미지 전처리 함수들
# ------------------------
def deskew(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    coords = np.column_stack(np.where(gray < 255))
    if coords.size == 0:
        return image
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45: angle += 90
    if abs(angle) < 1e-2:
        return image
    M = cv2.getRotationMatrix2D((gray.shape[1] // 2, gray.shape[0] // 2), angle, 1.0)
    return cv2.warpAffine(image, M, (gray.shape[1], gray.shape[0]), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

def remove_bolt_like_circles(image):
    output = image.copy()
    gray = cv2.cvtColor(output, cv2.COLOR_RGB2GRAY)
    blur = cv2.medianBlur(gray, 5)
    circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, dp=1.2, minDist=20,
                               param1=50, param2=40, minRadius=5, maxRadius=20)
    if circles is not None:
        for (x, y, r) in np.uint16(np.around(circles[0, :])):
            cv2.circle(output, (x, y), r + 2, (255, 255, 255), -1)
    return output

def enhance_contrast(image):
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_eq = clahe.apply(l)
    lab_eq = cv2.merge((l_eq, a, b))
    return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2RGB)

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

word_image_dir = project_root / f"data/word_images/{args.mode}"
word_label_dir = project_root / f"data/word_labels/{args.mode}"
word_image_dir.mkdir(parents=True, exist_ok=True)
word_label_dir.mkdir(parents=True, exist_ok=True)

craft_output_dir = Path("D:/Projects/CRNN_OCR_YGC/ocr/scripts/craft_output")
craft_output_dir.mkdir(exist_ok=True)

# ------------------------
# CRAFT 초기화
# ------------------------
craft = Craft(output_dir=str(craft_output_dir), crop_type="box", cuda=torch.cuda.is_available())

# ------------------------
# 단어/그룹 단위 분리 실행
# ------------------------
MIN_HEIGHT = 8
MIN_WIDTH = 8

for fname in os.listdir(image_dir):
    if not fname.lower().endswith(('.jpg', '.png')):
        continue

    try:
        fname_fixed = fname.encode('cp437').decode('utf-8')
    except (UnicodeEncodeError, UnicodeDecodeError):
        fname_fixed = fname
    fname = fname_fixed

    image_stem = os.path.splitext(fname)[0]
    base_name = image_stem if "_" in image_stem else image_stem + "_1"
    filename_prefix = sanitize_filename(base_name)

    label_path = label_dir / f"{image_stem}.txt"
    if not label_path.exists():
        print(f"❌ 라벨 없음: {label_path.name}")
        continue

    with open(label_path, "r", encoding="utf-8") as f:
        label_words = f.read().strip().split()

    if not label_words:
        print(f"❌ 라벨 내용 없음: {label_path.name}")
        continue

    image_path = image_dir / fname
    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)

    # 전처리 적용
    image_np = enhance_contrast(image_np)
    image_np = remove_bolt_like_circles(image_np)
    # image_np = deskew(image_np) # 기울기 보정 기능 비활성화

    debug_img_path = craft_output_dir / f"{filename_prefix}_preprocessed.jpg"
    cv2.imwrite(str(debug_img_path), cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))

    # CRAFT 실행
    prediction_result = craft.detect_text(image_np)
    boxes = prediction_result["boxes"]

    # --- 지능적 박스/라벨 처리 로직 ---

    # 특별 케이스: 라벨은 1개 단어인데, 박스가 2개 감지된 경우 -> 라벨을 박스 너비에 비례하여 분할
    if len(label_words) == 1 and len(boxes) == 2:
        print(f"ℹ️ {fname}: 1개 라벨, 2개 박스 감지. 라벨을 너비 비례로 분할합니다.")
        
        boxes = sorted(boxes, key=lambda b: b[0][0]) # 좌->우 정렬
        box1, box2 = boxes[0], boxes[1]

        width1 = get_box_width(box1)
        width2 = get_box_width(box2)
        total_width = width1 + width2

        if total_width > 0:
            label_string = label_words[0]
            split_point = round(len(label_string) * (width1 / total_width))
            split_point = max(1, min(len(label_string) - 1, split_point)) # 분할점이 최소 1글자는 되도록 보정

            label_words = [label_string[:split_point], label_string[split_point:]]
            print(f"    -> 분할된 라벨: {label_words}")

    # 노이즈 필터링: 실제 라벨 수보다 박스가 많이 감지되면, 가장 큰 박스들만 남김
    num_label_words = len(label_words)
    if len(boxes) > num_label_words:
        print(f"ℹ️ {fname}: 박스 {len(boxes)}개 > 라벨 {num_label_words}개. 가장 큰 박스 {num_label_words}개만 선택합니다.")
        boxes = sorted(boxes, key=get_box_area, reverse=True)
        boxes = boxes[:num_label_words]

    # 최종적으로 박스와 라벨을 x좌표 기준으로 정렬하여 매칭
    boxes = sorted(boxes, key=lambda b: b[0][0])

    if len(boxes) != num_label_words:
        print(f"❌ {fname} - 최종 박스 {len(boxes)}개 vs 라벨 단어 {num_label_words}개. 건너뜁니다.")
        continue

    for i, (box, word) in enumerate(zip(boxes, label_words)):
        x_min = int(min([pt[0] for pt in box]))
        x_max = int(max([pt[0] for pt in box]))
        y_min = int(min([pt[1] for pt in box]))
        y_max = int(max([pt[1] for pt in box]))

        y_min = max(0, y_min)
        y_max = min(image_np.shape[0], y_max)
        x_min = max(0, x_min)
        x_max = min(image_np.shape[1], x_max)

        cropped = image_np[y_min:y_max, x_min:x_max]

        if cropped.shape[0] < MIN_HEIGHT or cropped.shape[1] < MIN_WIDTH:
            print(f"ℹ️ {fname}의 {i}번째 박스가 너무 작아 건너뜁니다. (크기: {cropped.shape[1]}x{cropped.shape[0]})")
            continue

        word_img_path = word_image_dir / f"{filename_prefix}_{i}.jpg"
        word_lbl_path = word_label_dir / f"{filename_prefix}_{i}.txt"

        cv2.imwrite(str(word_img_path), cv2.cvtColor(cropped, cv2.COLOR_RGB2BGR))
        with open(word_lbl_path, "w", encoding="utf-8") as f:
            f.write(word)

    print(f"✅ {fname} → {len(boxes)}개 단어/그룹 분리 완료")

