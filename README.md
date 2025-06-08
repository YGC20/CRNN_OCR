# 차량 번호판 인식 OCR 프로젝트

이 프로젝트는 **대한민국 차량 번호판 인식**을 위한 YOLO + CRNN + CTC 기반 OCR 시스템입니다.  
AI Hub의 '자동차 번호판 영상 데이터셋'을 이용해 학습하였습니다.

---

## 📦 구성

- **번호판 검출**: YOLOv8
- **문자 인식**: CRNN + CTC Loss
- **데이터**: AI Hub 자동차 번호판 영상 데이터셋 (비공개)

---

## 🚀 실행 방법

### 1. 가상환경 준비
```bash
conda create -n plateocr python=3.9
conda activate plateocr
pip install -r requirements.txt
