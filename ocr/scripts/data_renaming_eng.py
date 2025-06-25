import os
import shutil
from pathlib import Path
from jamo import h2j

# 자모 → 로마자 (유니코드 조합형 자모용: U+1100~)
jamo_to_roman = {
    # 초성 (U+1100~)
    'ᄀ': 'g',  'ᄁ': 'kk', 'ᄂ': 'n',  'ᄃ': 'd',  'ᄄ': 'tt',
    'ᄅ': 'r',  'ᄆ': 'm',  'ᄇ': 'b',  'ᄈ': 'pp', 'ᄉ': 's',
    'ᄊ': 'ss', 'ᄋ': '',   'ᄌ': 'j',  'ᄍ': 'jj', 'ᄎ': 'ch',
    'ᄏ': 'k',  'ᄐ': 't',  'ᄑ': 'p',  'ᄒ': 'h',

    # 중성 (U+1160~)
    'ᅡ': 'a',  'ᅢ': 'ae',  'ᅣ': 'ya',  'ᅤ': 'yae',
    'ᅥ': 'eo', 'ᅦ': 'e',   'ᅧ': 'yeo', 'ᅨ': 'ye',
    'ᅩ': 'o',  'ᅪ': 'wa',  'ᅫ': 'wae', 'ᅬ': 'oe',
    'ᅭ': 'yo', 'ᅮ': 'u',   'ᅯ': 'wo',  'ᅰ': 'we',
    'ᅱ': 'wi', 'ᅲ': 'yu',  'ᅳ': 'eu',  'ᅴ': 'ui', 'ᅵ': 'i',

    # 종성 (U+11A8~)
    'ᆨ': 'g',  'ᆩ': 'kk', 'ᆪ': 'gs', 'ᆫ': 'n',  'ᆬ': 'nj',
    'ᆭ': 'nh', 'ᆮ': 'd',  'ᆯ': 'l',  'ᆰ': 'lg', 'ᆱ': 'lm',
    'ᆲ': 'lb', 'ᆳ': 'ls', 'ᆴ': 'lt', 'ᆵ': 'lp', 'ᆶ': 'lh',
    'ᆷ': 'm',  'ᆸ': 'b',  'ᆹ': 'bs', 'ᆺ': 's',  'ᆻ': 'ss',
    'ᆼ': 'ng', 'ᆽ': 'j',  'ᆾ': 'ch', 'ᆿ': 'k',  'ᇀ': 't',
    'ᇁ': 'p',  'ᇂ': 'h'
}

def convert_name(text):
    result = []
    for char in text:
        if '가' <= char <= '힣':
            jamos = list(h2j(char))
            romanized = []
            for j in jamos:
                rom = jamo_to_roman.get(j, 'xx')
                if rom == 'xx':
                    print(f"⚠️ 변환 실패: '{char}'의 자모 '{j}' → 'xx'")
                romanized.append(rom)
            result.append(''.join(romanized))
        else:
            result.append(char)
    return ''.join(result)

def rename_and_copy_by_order(image_dir, label_dir, out_image_dir, out_label_dir):
    out_image_dir.mkdir(parents=True, exist_ok=True)
    out_label_dir.mkdir(parents=True, exist_ok=True)

    image_list = sorted([f for f in Path(image_dir).iterdir() if f.suffix.lower() in ['.jpg', '.png']])
    label_list = sorted([f for f in Path(label_dir).glob("*.txt")])

    for label_path, image_path in zip(label_list, image_list):
        with open(label_path, encoding='utf-8') as f:
            label_text = f.read().strip()
        new_name = convert_name(label_text)

        # 복사
        shutil.copy(image_path, out_image_dir / f"{new_name}{image_path.suffix}")
        shutil.copy(label_path, out_label_dir / f"{new_name}.txt")

        print(f"✅ {image_path.name} → {new_name}{image_path.suffix}")

# 경로 설정
project_root = Path(__file__).resolve().parent.parent

rename_and_copy_by_order(
    image_dir = project_root / "data/images/train",
    label_dir = project_root / "data/labels/train",
    out_image_dir = project_root / "data/eng_images/train",
    out_label_dir = project_root / "data/eng_labels/train"
)

rename_and_copy_by_order(
    image_dir = project_root / "data/images/val",
    label_dir = project_root / "data/labels/val",
    out_image_dir = project_root / "data/eng_images/val",
    out_label_dir = project_root / "data/eng_labels/val"
)
