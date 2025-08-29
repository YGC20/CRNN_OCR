import os
from pathlib import Path
from PIL import Image
import random

def verify_dataset():
    """
    CRNN OCR 가상 데이터셋의 무결성을 검증하는 스크립트.
    1. 학습/검증 데이터의 이미지-라벨 개수 일치 여부 확인
    2. 이미지-라벨 파일명 짝이 맞는지 확인
    3. 샘플 이미지 파일과 라벨 파일을 직접 열어 유효성 확인
    """
    project_root = Path(__file__).resolve().parent
    base_data_path = project_root / "ocr/data"

    paths_to_check = {
        "train": {
            "images": base_data_path / "virtual_data/images",
            "labels": base_data_path / "virtual_data/labels",
        },
        "validation": {
            "images": base_data_path / "virtual_val_data/images",
            "labels": base_data_path / "virtual_val_data/labels",
        }
    }

    print("="*50)
    print("데이터셋 무결성 검증을 시작합니다.")
    print("="*50)

    all_ok = True

    for phase, paths in paths_to_check.items():
        print(f"\n--- {phase.upper()} 데이터셋 검증 ---")
        image_path = paths["images"]
        label_path = paths["labels"]

        if not image_path.exists() or not label_path.exists():
            print(f"[오류] {phase} 데이터셋 경로를 찾을 수 없습니다:")
            print(f"  - 이미지 경로: {image_path}")
            print(f"  - 라벨 경로: {label_path}")
            all_ok = False
            continue

        image_files = os.listdir(image_path)
        label_files = os.listdir(label_path)

        # 1. 파일 개수 확인
        print(f"  [1] 파일 개수 확인:")
        print(f"    - 이미지 파일 수: {len(image_files)}")
        print(f"    - 라벨 파일 수: {len(label_files)}")
        if len(image_files) != len(label_files):
            print("    - [오류] 이미지와 라벨 파일의 개수가 일치하지 않습니다!")
            all_ok = False
        else:
            print("    - [성공] 파일 개수가 일치합니다.")

        # 2. 파일명 짝 확인
        print(f"\n  [2] 파일명 일치 여부 확인:")
        image_basenames = {Path(f).stem for f in image_files}
        label_basenames = {Path(f).stem for f in label_files}

        unmatched_images = image_basenames - label_basenames
        unmatched_labels = label_basenames - image_basenames

        if not unmatched_images and not unmatched_labels:
            print("    - [성공] 모든 이미지와 라벨 파일명이 1:1로 일치합니다.")
        else:
            all_ok = False
            if unmatched_images:
                print(f"    - [오류] 라벨이 없는 이미지 파일 {len(unmatched_images)}개 발견.")
                # 예시 출력
                print(f"      (예시: {list(unmatched_images)[:3]})")
            if unmatched_labels:
                print(f"    - [오류] 이미지가 없는 라벨 파일 {len(unmatched_labels)}개 발견.")
                print(f"      (예시: {list(unmatched_labels)[:3]})")

        # 3. 샘플 파일 유효성 검사
        print(f"\n  [3] 샘플 파일 유효성 검사:")
        if not image_files:
            print("    - 이미지가 없어 샘플 검사를 건너뜁니다.")
            continue
            
        num_samples = min(5, len(image_files))
        sample_files = random.sample(image_files, num_samples)
        
        sample_ok_count = 0
        for img_file in sample_files:
            basename = Path(img_file).stem
            label_file = basename + ".txt"
            
            img_full_path = image_path / img_file
            lbl_full_path = label_path / label_file

            try:
                # 이미지 파일 열기 시도
                with Image.open(img_full_path) as img:
                    img.verify() # 이미지 데이터 유효성 검사

                # 라벨 파일 열기 및 읽기 시도 (UTF-8)
                with open(lbl_full_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if not content:
                        raise ValueError("라벨 파일 내용이 비어있습니다.")

                print(f"    - [성공] 샘플 '{basename}' (라벨: '{content}') 파일이 유효합니다.")
                sample_ok_count += 1
            except Exception as e:
                print(f"    - [오류] 샘플 '{basename}' 처리 중 오류 발생: {e}")
                all_ok = False
    
    print("\n" + "="*50)
    if all_ok:
        print("✅ 최종 결과: 모든 데이터셋 검증을 통과했습니다.")
    else:
        print("❌ 최종 결과: 데이터셋에서 하나 이상의 문제가 발견되었습니다.")
    print("="*50)


if __name__ == "__main__":
    verify_dataset()
