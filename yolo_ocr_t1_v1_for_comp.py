# ───────────────────────────────────────────────────────────────────────────────
# 파일명: yolo_ocr_t1_v1_for_comp.py
#   • 전처리 - YOLO - OCR
#   • conf_thres: 0.25
#   • yolo_ocr_t2.py와 동일한 출력 형식으로 변경
# ───────────────────────────────────────────────────────────────────────────────

import sys
import os
import re
import cv2
import torch
import numpy as np
import pandas as pd
from pathlib import Path

# ──────────── (1) project_root 및 yolov5 모듈 검색 경로 추가 ────────────
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root / "yolov5"))
sys.path.insert(1, str(project_root))
# ────────────────────────────────────────────────────────────────────────────

from PIL import Image
from torchvision import transforms
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.augmentations import letterbox
from yolov5.utils.general import non_max_suppression, scale_boxes
from ocr.model.crnn import CRNN
from ocr.utils.label_encoder import LabelEncoder

# -------------------- 전처리 및 OCR 관련 유틸 함수 --------------------
def remove_bolt_like_circles(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray, 5)
    circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, dp=1.2, minDist=20,
                               param1=50, param2=30, minRadius=5, maxRadius=15)
    if circles is not None:
        circles = np.uint16(np.around(circles[0, :]))
        for (x, y, r) in circles:
            cv2.circle(image, (x, y), r + 2, (255, 255, 255), -1)
    return image

def correct_skew(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)
    if lines is None:
        return image
    angles = [np.degrees(theta - np.pi / 2) for rho, theta in lines[:, 0]]
    median_angle = np.median(angles)
    (h, w) = image.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), median_angle, 1.0)
    return cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

def preprocess_enhanced(image):
    img = image.copy()
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l2 = cv2.createCLAHE(2.0, (8, 8)).apply(l)
    img = cv2.merge((l2, a, b))
    img = cv2.cvtColor(img, cv2.COLOR_LAB2BGR)
    gaussian = cv2.GaussianBlur(img, (0, 0), 3)
    img = cv2.addWeighted(img, 1.5, gaussian, -0.5, 0)
    img = cv2.bilateralFilter(img, 9, 75, 75)
    img = remove_bolt_like_circles(img)
    return img

def resize_with_padding(image, target_size=(128, 32)):
    h, w = image.shape[:2]
    target_w, target_h = target_size
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(image, (new_w, new_h))
    padded = np.ones((target_h, target_w), dtype=np.uint8) * 255
    pad_left = (target_w - new_w) // 2
    pad_top = (target_h - new_h) // 2
    padded[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = resized
    return padded

def preprocess_image(image, target_width=128, target_height=32):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = resize_with_padding(image, (target_width, target_height))
    image = image.astype(np.float32) / 255.0
    image = torch.from_numpy(image).unsqueeze(0).unsqueeze(0)
    return image

def decode_prediction(preds, label_encoder):
    preds = preds.permute(1, 0, 2)
    out = torch.nn.functional.log_softmax(preds, dim=2)
    _, out_best = out.max(2)
    out_best = out_best.transpose(1, 0).contiguous().view(-1)
    pred_standard = label_encoder.decode_ctc_standard(out_best.cpu().numpy())
    return pred_standard

def crnn_ocr(image, crnn_model, label_encoder, device):
    input_tensor = preprocess_image(image).to(device)
    with torch.no_grad():
        preds = crnn_model(input_tensor)
    text = decode_prediction(preds, label_encoder)
    return text

def load_crnn_model(model_path, num_classes, device):
    model = CRNN(img_height=32, num_channels=1, num_classes=num_classes, hidden_size=512)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def parse_ground_truth(filename):
    mapping = {
        'beo': '버', 'ga': '가', 'na': '나', 'da': '다', 'deo': '더', 'do': '도', 
        'du': '두', 'eo': '어', 'seo': '서', 'bo': '보', 'bu': '부', 'no': '노',
        'o': '오', 'ju': '주', 'meo': '머', 'ho': '호', 'ha': '하'
    }
    stem = Path(filename).stem.lower()
    # Remove suffixes like _1, _2
    stem = re.sub(r'_[0-9]+$', '', stem)
    for rom, kor in mapping.items():
        stem = stem.replace(rom, kor)
    return stem.upper()

# -------------------- 메인 루틴 --------------------
def main():
    yolo_model_path = 'bestModel/yoloBestModel.pt'
    crnn_model_path = 'bestModel/ocrBestModel_142.pth'
    test_image_dir = 'images/test_images_kr'
    output_dir = 'images/ocr_results'
    os.makedirs(output_dir, exist_ok=True)

    label_encoder = LabelEncoder()
    charset = label_encoder.get_charset()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    yolo_model = DetectMultiBackend(yolo_model_path, device=device)
    crnn_model = load_crnn_model(crnn_model_path, num_classes=len(charset)+1, device=device)

    num_runs = 50
    results = {}

    for filename in os.listdir(test_image_dir):
        if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        
        gt = parse_ground_truth(filename)
        if not gt: continue

        image_path = os.path.join(test_image_dir, filename)
        original_image = cv2.imread(image_path)
        if original_image is None:
            continue

        enhanced_img = preprocess_enhanced(original_image)
        image_resized = letterbox(enhanced_img, new_shape=640)[0]
        image_resized = image_resized[:, :, ::-1].transpose(2, 0, 1)
        image_resized = np.ascontiguousarray(image_resized, dtype=np.float32) / 255.0
        image_tensor = torch.from_numpy(image_resized).unsqueeze(0).to(device)

        with torch.no_grad():
            pred = yolo_model(image_tensor)
        pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45)[0]

        run_preds = []
        if pred is not None and len(pred):
            # Use the first detected box
            box = pred[0]
            x1, y1, x2, y2 = map(int, scale_boxes(image_tensor.shape[2:], box[:4], original_image.shape).round())
            cropped = original_image[y1:y2, x1:x2]
            cropped = correct_skew(cropped)
            
            for _ in range(num_runs):
                text = crnn_ocr(cropped, crnn_model, label_encoder, device)
                run_preds.append(text)
        else:
            run_preds = [''] * num_runs
            
        results[gt] = run_preds

    df = pd.DataFrame.from_dict(
        results,
        orient='index',
        columns=[str(i+1) for i in range(num_runs)]
    )

    accuracies = []
    total = len(df)
    if total > 0:
        for col in df.columns:
            correct = (df[col] == df.index).sum()
            accuracies.append(f"{correct/total*100:.2f}%")
    else:
        accuracies = ['0.00%'] * num_runs
        
    df.loc['정답률'] = accuracies

    csv_path = os.path.join(output_dir, 'comp_results_t1_v1.csv')
    df.to_csv(csv_path, index_label='ground_truth', encoding='utf-8-sig')
    print(f"✔ v1 방식의 OCR 결과를 CSV로 저장했습니다: {csv_path}")

if __name__ == '__main__':
    main()
