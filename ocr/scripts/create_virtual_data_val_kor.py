
import os
import random
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# --- 설정 ---
# (학습 데이터 생성 스크립트에서 경로와 개수만 변경)
OUTPUT_DIR = os.path.join("ocr", "data", "virtual_val_data_kor")
NUM_IMAGES = 11500  # 학습 데이터의 10%
IMAGE_WIDTH = 200
IMAGE_HEIGHT = 50
FONT_PATH = "C:/Windows/Fonts/malgun.ttf"  # 맑은 고딕 또는 다른 한글 지원 폰트
MIN_FONT_SIZE = 30
MAX_FONT_SIZE = 45

# --- 문자셋 정의 (영문 제외) ---
# 1. 완성형 한글 (2350자)
HANGUL_CHARS = [chr(0xAC00 + i) for i in range(11172)]

# 2. 한글 자모 (51자)
initial_consonants = [chr(i) for i in range(0x1100, 0x1113)]
vowels = [chr(i) for i in range(0x1161, 0x1176)]
final_consonants = [chr(i) for i in range(0x11A8, 0x11C3)]
JAMO_CHARS = initial_consonants + vowels + final_consonants

# 3. 숫자 (10자)
NUMBER_CHARS = "0123456789"

# 4. 특수문자 (8자)
SPECIAL_CHARS = ".,!@#$%^&"

# 최종 문자셋 (한글 + 자모 + 숫자 + 특수문자)
KOR_CHARSET = "".join(sorted(list(set(HANGUL_CHARS + JAMO_CHARS + list(NUMBER_CHARS) + list(SPECIAL_CHARS)))))

# --- 이미지 생성 ---
def generate_image(char, font):
    """지정된 문자로 이미지를 생성합니다."""
    image = Image.new('L', (IMAGE_WIDTH, IMAGE_HEIGHT), color=255)
    draw = ImageDraw.Draw(image)
    
    try:
        bbox = draw.textbbox((0, 0), char, font=font)
        char_width = bbox[2] - bbox[0]
        char_height = bbox[3] - bbox[1]
    except AttributeError:
        char_width, char_height = draw.textsize(char, font=font)

    x = random.randint(0, max(0, IMAGE_WIDTH - char_width - 10))
    y = random.randint(0, max(0, IMAGE_HEIGHT - char_height - 10))
    
    draw.text((x, y), char, fill=0, font=font)
    return image

def main():
    """메인 실행 함수"""
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created directory: {OUTPUT_DIR}")

    label_file_path = os.path.join(OUTPUT_DIR, "labels.txt")
    with open(label_file_path, "w", encoding="utf-8") as f:
        pass

    num_chars = len(KOR_CHARSET)
    for i in range(NUM_IMAGES):
        char_to_draw = KOR_CHARSET[i % num_chars]
        
        font_size = random.randint(MIN_FONT_SIZE, MAX_FONT_SIZE)
        font = ImageFont.truetype(FONT_PATH, font_size)
        
        image = generate_image(char_to_draw, font)
        
        image_filename = f"image_{i+1:06d}.png"
        image_path = os.path.join(OUTPUT_DIR, image_filename)
        image.save(image_path)
        
        with open(label_file_path, "a", encoding="utf-8") as f:
            f.write(f"{image_filename}\t{char_to_draw}\n")
            
        if (i + 1) % 1000 == 0:
            print(f"Generated {i + 1}/{NUM_IMAGES} validation images...")

    print(f"Successfully generated {NUM_IMAGES} validation images and labels.txt in {OUTPUT_DIR}")
    print(f"Total characters in charset: {len(KOR_CHARSET)}")

if __name__ == "__main__":
    main()
