import cv2
import torch
import numpy as np
from craft_text_detector import Craft
from pathlib import Path
import argparse

# ------------------------
# 이미지 전처리 함수
# ------------------------
def preprocess_image(image_path):
    """이미지를 불러와 기울기 보정 및 대비 향상을 적용합니다."""
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"이미지를 찾을 수 없습니다: {image_path}")
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 1. 기울기 보정 (Deskew)
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    coords = np.column_stack(np.where(gray < 255))
    if coords.size > 0:
        # 최소 영역 사각형을 찾음 (중심점, (너비, 높이), 각도)
        rect = cv2.minAreaRect(coords)
        box_w, box_h = rect[1]
        angle = rect[2]

        # 박스의 너비가 높이보다 작으면 (세로로 서 있는 경우) 각도를 조정
        if box_w < box_h:
            angle += 90

        # 너무 작은 기울기는 무시
        if abs(angle) > 1:
            h, w = image_rgb.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            image_rgb = cv2.warpAffine(image_rgb, M, (w, h), 
                                     flags=cv2.INTER_CUBIC, 
                                     borderMode=cv2.BORDER_REPLICATE)

    # 2. 대비 향상 (CLAHE)
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_eq = clahe.apply(l)
    lab_eq = cv2.merge((l_eq, a, b))
    image_processed = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2RGB)
    
    return image_processed

# ------------------------
# 메인 실행 함수
# ------------------------
def process_single_image(image_path, craft_model, output_dir):
    """단일 이미지에 대한 전처리 및 CRAFT 감지, 저장을 수행합니다."""
    try:
        print(f"\n▶ 처리 시작: {image_path.name}")
        # 이미지 전처리
        preprocessed_img = preprocess_image(image_path)

        # CRAFT로 텍스트 영역 감지
        prediction_result = craft_model.detect_text(preprocessed_img)
        boxes = prediction_result["boxes"]

        if len(boxes) == 0:
            print(f"  - 텍스트 영역을 찾지 못했습니다.")
            return

        # 감지된 박스를 x좌표 기준으로 정렬
        boxes = sorted(boxes, key=lambda b: b[0][0])
        print(f"  - {len(boxes)}개의 텍스트 영역 감지. 이미지 저장 중...")

        # 각 박스 영역을 잘라내어 저장
        for i, box in enumerate(boxes):
            x_min = int(min([pt[0] for pt in box]))
            x_max = int(max([pt[0] for pt in box]))
            y_min = int(min([pt[1] for pt in box]))
            y_max = int(max([pt[1] for pt in box]))

            y_min = max(0, y_min)
            y_max = min(preprocessed_img.shape[0], y_max)
            x_min = max(0, x_min)
            x_max = min(preprocessed_img.shape[1], x_max)

            cropped_img = preprocessed_img[y_min:y_max, x_min:x_max]

            if cropped_img.size == 0:
                continue

            output_filename = output_dir / f"{image_path.stem}_crop_{i:02d}.png"
            cv2.imwrite(str(output_filename), cv2.cvtColor(cropped_img, cv2.COLOR_RGB2BGR))
        
        print(f"  - ✓ 저장 완료")

    except Exception as e:
        print(f"  - ❌ 오류 발생: {e}")

def main(args):
    # 경로 설정
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # CRAFT 모델은 한 번만 초기화
    print("CRAFT 모델을 초기화합니다...")
    craft = Craft(crop_type="box", cuda=torch.cuda.is_available())

    if input_path.is_dir():
        print(f"입력 경로가 폴더입니다. '{input_path}' 내의 모든 이미지를 처리합니다.")
        image_files = list(input_path.glob('*.[jJ][pP][gG]')) + \
                      list(input_path.glob('*.[jJ][pP][eE][gG]')) + \
                      list(input_path.glob('*.[pP][nN][gG]'))
        
        if not image_files:
            print("처리할 이미지가 폴더에 없습니다.")
            return
            
        for image_file in image_files:
            process_single_image(image_file, craft, output_dir)

    elif input_path.is_file():
        print(f"입력 경로가 단일 파일입니다. '{input_path}' 파일을 처리합니다.")
        process_single_image(input_path, craft, output_dir)
    else:
        print(f"오류: 입력 경로를 찾을 수 없습니다: {input_path}")
        return

    print(f"\n✅ 모든 작업 완료! 결과가 {output_dir}에 저장되었습니다.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="이미지(또는 폴더)를 전처리하고 CRAFT로 텍스트 영역을 잘라냅니다.")
    parser.add_argument('--input', type=str, required=True, help="처리할 이미지 파일 또는 폴더 경로")
    parser.add_argument('--output_dir', type=str, default="output_crops", help="잘라낸 이미지를 저장할 디렉토리")
    
    args = parser.parse_args()
    main(args)
