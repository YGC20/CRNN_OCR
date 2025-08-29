import os
import random
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
import shutil

# --- 설정 ---

# 1. 조합할 한글 자모 정의 (주석 처리 - 단순화를 위해 사용 안 함)
# CHOSUNG_LIST = ['ㄱ', 'ㄴ', 'ㄷ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅅ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']
# JOONGSUNG_LIST = ['ㅛ', 'ㅕ', 'ㅑ', 'ㅐ', 'ㅔ', 'ㅗ', 'ㅓ', 'ㅏ', 'ㅣ', 'ㅠ', 'ㅜ', 'ㅡ']

# 2. 유니코드 기반 한글 조합 함수 (주석 처리 - 단순화를 위해 사용 안 함)
# def combine_hangul(chosung, joongsung):
#     CHOSUNG_MAP = {'ㄱ': 0, 'ㄲ': 1, 'ㄴ': 2, 'ㄷ': 3, 'ㄸ': 4, 'ㄹ': 5, 'ㅁ': 6, 'ㅂ': 7, 'ㅃ': 8, 'ㅅ': 9, 'ㅆ': 10, 'ㅇ': 11, 'ㅈ': 12, 'ㅉ': 13, 'ㅊ': 14, 'ㅋ': 15, 'ㅌ': 16, 'ㅍ': 17, 'ㅎ': 18}
#     JOONGSUNG_MAP = {'ㅏ': 0, 'ㅐ': 1, 'ㅑ': 2, 'ㅒ': 3, 'ㅓ': 4, 'ㅔ': 5, 'ㅕ': 6, 'ㅖ': 7, 'ㅗ': 8, 'ㅘ': 9, 'ㅫ': 10, 'ㅚ': 11, 'ㅛ': 12, 'ㅜ': 13, 'ㅝ': 14, 'ㅞ': 15, 'ㅟ': 16, 'ㅠ': 17, 'ㅡ': 18, 'ㅢ': 19, 'ㅣ': 20}
#     chosung_idx = CHOSUNG_MAP.get(chosung)
#     joongsung_idx = JOONGSUNG_MAP.get(joongsung)

#     if chosung_idx is None or joongsung_idx is None:
#         return None

#     return chr(0xAC00 + chosung_idx * 21 * 28 + joongsung_idx * 28)

# 3. 전체 문자셋 생성 (단순화된 문자셋 사용)
# combined_hangul = [combine_hangul(c, j) for c in CHOSUNG_LIST for j in JOONGSUNG_LIST]
# JAMO_CHARS = CHOSUNG_LIST + JOONGSUNG_LIST
ENGLISH_CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
NUMBER_CHARS = "0123456789"
TOTAL_CHARS = list(ENGLISH_CHARS) + list(NUMBER_CHARS)

# 이미지 설정
IMG_SIZE = (64, 64)
BACKGROUND_COLOR = (255, 255, 255)
NUM_VARIATIONS_PER_CHAR = 50  # 검증용 데이터는 양을 줄임

# 경로 설정
SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "ocr/data/virtual_val_data" # 저장 위치 변경
IMAGE_DIR = OUTPUT_DIR / "images"
LABEL_DIR = OUTPUT_DIR / "labels"

# 폰트 설정 (다중 폰트 지원)
FONT_PATHS = [
    "C:/Windows/Fonts/malgun.ttf",
    "C:/Windows/Fonts/gulim.ttc",
    "C:/Windows/Fonts/batang.ttc",
    "C:/Windows/Fonts/H2GTRE.TTF",
    "C:/Windows/Fonts/H2GTRM.TTF",
]

AVAILABLE_FONTS = [p for p in FONT_PATHS if Path(p).exists()]
if not AVAILABLE_FONTS:
    print("사용 가능한 폰트를 찾을 수 없습니다.")
    exit()
print(f"사용 가능한 폰트: {AVAILABLE_FONTS}")

# --- 이미지 증강 함수들 (훈련용과 동일) ---
def add_noise(image, strength=0.5):
    h, w, c = image.shape
    mean = 0
    var = (strength * 255 * 0.5)**2
    sigma = var**0.5
    gauss = np.random.normal(mean, sigma, (h, w, c))
    noisy = np.clip(image + gauss, 0, 255)
    return noisy.astype(np.uint8)

def apply_blur(image, strength=0.5):
    ksize = int(strength * 10) * 2 + 1
    return cv2.GaussianBlur(image, (ksize, ksize), 0)

def change_brightness_contrast(image, alpha_range=(0.6, 1.4), beta_range=(-50, 50)):
    alpha = random.uniform(*alpha_range)
    beta = random.uniform(*beta_range)
    return np.clip(alpha * image + beta, 0, 255).astype(np.uint8)

def apply_geometric_transform(image):
    h, w = image.shape[:2]
    angle = random.uniform(-10, 10)
    scale = random.uniform(0.8, 1.1)
    tx = random.uniform(-w * 0.1, w * 0.1)
    ty = random.uniform(-h * 0.1, h * 0.1)
    M_rot = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
    transformed = cv2.warpAffine(image, M_rot, (w, h), borderValue=BACKGROUND_COLOR)
    M_trans = np.float32([[1, 0, tx], [0, 1, ty]])
    transformed = cv2.warpAffine(transformed, M_trans, (w, h), borderValue=BACKGROUND_COLOR)
    return transformed

# --- 메인 생성 함수 ---
def generate_data():
    if OUTPUT_DIR.exists():
        print(f"기존 폴더를 삭제합니다: {OUTPUT_DIR}")
        shutil.rmtree(OUTPUT_DIR)
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    LABEL_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[검증용 데이터 생성] 총 {len(TOTAL_CHARS)}개의 문자에 대해 각각 {NUM_VARIATIONS_PER_CHAR}개의 이미지를 생성합니다.")
    print(f"예상 생성 파일 수: {len(TOTAL_CHARS) * NUM_VARIATIONS_PER_CHAR * 2} 개")
    print(f"결과 저장 위치: {OUTPUT_DIR}")

    fonts = [ImageFont.truetype(font_path, size=40) for font_path in AVAILABLE_FONTS]

    total_count = 0
    for char in TOTAL_CHARS:
        if char is None: continue
        char_count = 0
        for i in range(NUM_VARIATIONS_PER_CHAR):
            font = random.choice(fonts)
            image_pil = Image.new("RGB", IMG_SIZE, BACKGROUND_COLOR)
            draw = ImageDraw.Draw(image_pil)
            
            try:
                bbox = draw.textbbox((0, 0), char, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
            except AttributeError:
                text_width, text_height = draw.textsize(char, font=font)

            x = (IMG_SIZE[0] - text_width) / 2
            y = (IMG_SIZE[1] - text_height) / 2
            draw.text((x, y), char, font=font, fill=(0, 0, 0))

            image_cv = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
            augmented_img = apply_geometric_transform(image_cv)

            if random.random() < 0.5: augmented_img = change_brightness_contrast(augmented_img)
            if random.random() < 0.5: augmented_img = apply_blur(augmented_img, strength=random.random())
            if random.random() < 0.5: augmented_img = add_noise(augmented_img, strength=random.random())

            safe_char = char if char.isalnum() else f"c{ord(char)}"
            filename = f"virtual_val_{safe_char}_{i:03d}"
            img_path = IMAGE_DIR / f"{filename}.png"
            lbl_path = LABEL_DIR / f"{filename}.txt"

            try:
                result, buffer = cv2.imencode('.png', augmented_img)
                if not result: continue
                with open(img_path, 'wb') as f: f.write(buffer)
                with open(lbl_path, 'w', encoding='utf-8') as f: f.write(char)
                char_count += 1
            except Exception as e:
                print(f"\n[예외 발생] '{char}' 저장 실패: {e}")
                continue
        
        total_count += char_count
        print(f"  - 문자 '{char}' 생성 완료 ({char_count}개)", end='\r')

    print("\n" + "-"*50)
    print(f"[SUCCESS] Total {total_count} validation data have been created.")
    print("-"*50)

if __name__ == "__main__":
    generate_data()
