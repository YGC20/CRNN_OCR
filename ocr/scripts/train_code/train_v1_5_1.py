# train_v1_5_1.py - 가상 데이터셋으로 CRNN + CTC OCR 학습 스크립트

# 기본 라이브러리 및 PyTorch 관련 모듈 불러오기
import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import wandb    # Weights & Biases: 실험 로깅 및 시각화 도구
import random
import numpy as np
from torch.utils.data import DataLoader
from collections import Counter
from torchvision import transforms
from pathlib import Path

# 사용자 정의 모듈 불러오기
from utils.dataset import OCRDataset # 이미지/레이블 데이터셋 클래스
from utils.label_encoder import LabelEncoder    # 텍스트를 숫자로 인코딩/디코딩 클래스
from model.crnn import CRNN # CRNN 모델 정의

# --- 가상 데이터 생성 스크립트에서 문자셋 가져오기 ---
# 1. 조합할 한글 자모 정의 (주석 처리 - 단순화를 위해 사용 안 함)
# CHOSUNG_LIST = ['ㄱ', 'ㄴ', 'ㄷ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅅ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']
# JOONGSUNG_LIST = ['ㅛ', 'ㅕ', 'ㅑ', 'ㅐ', 'ㅔ', 'ㅗ', 'ㅓ', 'ㅏ', 'ㅣ', 'ㅠ', 'ㅜ', 'ㅡ']

# 2. 유니코드 기반 한글 조합 함수 (주석 처리 - 단순화를 위해 사용 안 함)
# def combine_hangul(chosung, joongsung):
#     CHOSUNG_MAP = {'ㄱ': 0, 'ㄲ': 1, 'ㄴ': 2, 'ㄷ': 3, 'ㄸ': 4, 'ㄹ': 5, 'ㅁ': 6, 'ㅂ': 7, 'ㅃ': 8, 'ㅅ': 9, 'ㅆ': 10, 'ㅇ': 11, 'ㅈ': 12, 'ㅉ': 13, 'ㅊ': 14, 'ㅋ': 15, 'ㅌ': 16, 'ㅍ': 17, 'ㅎ': 18}
#     JOONGSUNG_MAP = {'ㅏ': 0, 'ㅐ': 1, 'ㅑ': 2, 'ㅒ': 3, 'ㅓ': 4, 'ㅔ': 5, 'ㅕ': 6, 'ㅖ': 7, 'ㅗ': 8, 'ㅘ': 9, 'ㅫ': 10, 'ㅚ': 11, 'ㅛ': 12, 'ㅜ': 13, 'ㅝ': 14, 'ㅞ': 15, 'ㅟ': 16, 'ㅠ': 17, 'ㅡ': 18, 'ㅢ': 19, 'ㅣ': 20}
#     chosung_idx = CHOSUNG_MAP.get(chosung)
#     joongsung_idx = JOONGSUNG_MAP.get(joongsung)
#     if chosung_idx is None or joongsung_idx is None: return None
#     return chr(0xAC00 + chosung_idx * 21 * 28 + joongsung_idx * 28)

# 3. 전체 문자셋 생성 (한글, 영문, 숫자 포함)
KOREAN_CHARS = '가강거경계고관광구금기김나남너노누다대더도동두등라러로루리마머명모무문미바버배뱌버보부북사산서소수아악안양어연영오용우울원육이인자작저전조주중지차천처초추충카커코쿠타터토투파평퍼포푸하허호홀후후히'
ENGLISH_CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ" # 대문자 영어만 사용
NUMBER_CHARS = "0123456789"
VIRTUAL_CHARSET = KOREAN_CHARS + ENGLISH_CHARS + NUMBER_CHARS
# --- 문자셋 정의 끝 ---


# 모델 저장 경로 수정
ocr_dir = Path(__file__).resolve().parent
project_root = ocr_dir.parent
bestmodel_dir = project_root / "bestModel"
bestmodel_dir.mkdir(exist_ok=True)
best_path = bestmodel_dir / "ocrBestModel_151.pth"

# ---------------------------
# 문자 단위 정확도 계산 함수
# ---------------------------
def char_accuracy(gt, pred):
    if not gt or not pred:
        return 0.0
    match = sum(g == p for g, p in zip(gt, pred))
    return match / max(len(gt), len(pred)) * 100

# ---------------------------
# Beam Search 디코딩 함수
# ---------------------------
def decode_beam_search(logits, encoder, beam_width=3):
    log_probs = logits.log_softmax(2).cpu().detach().numpy()
    log_probs = np.transpose(log_probs, (1, 0, 2))

    results = []
    for seq in log_probs:
        paths = [([], 0)]
        for t in seq:
            new_paths = []
            topk = np.argsort(t)[-beam_width:][::-1]
            for path, score in paths:
                for k in topk:
                    new_path = path + [k]
                    new_score = score + t[k]
                    new_paths.append((new_path, new_score))
            paths = sorted(new_paths, key=lambda x: x[1], reverse=True)[:beam_width]
        best_path = paths[0][0]
        decoded = encoder.decode_ctc_standard(best_path)
        results.append(decoded)
    return results

# ---------------------------
# 모델 평가 함수 (Beam Search 기반)
# ---------------------------
def evaluate(model, val_loader, encoder, criterion, device):
    model.eval()
    total = 0
    correct = 0
    val_loss = 0.0

    with torch.no_grad():
        for images, texts in val_loader:
            images = images.to(device)
            outputs = model(images) # log_softmax는 criterion 내부에서 처리

            targets = [torch.tensor(encoder.encode(t), dtype=torch.long) for t in texts]
            target_lengths = torch.tensor([len(t) for t in targets], dtype=torch.long)
            targets_concat = torch.cat(targets).to(device)
            input_lengths = torch.full(size=(images.size(0),), fill_value=outputs.size(0), dtype=torch.long)

            loss = criterion(outputs.log_softmax(2), targets_concat, input_lengths, target_lengths)
            val_loss += loss.item()

            decoded_texts = decode_beam_search(outputs, encoder, beam_width=5)
            for pred_text, gt_text in zip(decoded_texts, texts):
                total += 1
                if pred_text == gt_text:
                    correct += 1

    acc = (correct / total * 100) if total > 0 else 0
    avg_loss = (val_loss / len(val_loader)) if len(val_loader) > 0 else 0
    return acc, avg_loss

# ---------------------------
# custom collate_fn 함수
# ---------------------------
def custom_collate_fn(batch):
    images, texts = zip(*batch)
    images = torch.stack(images, dim=0)
    return images, list(texts)

# ---------------------------
# 메인 학습 루틴
# ---------------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # wandb 초기화
    wandb.init(
        project="crnn-ocr",
        name="train_v1.5.1",
        dir=str(ocr_dir / "wandb_logs"),
        config={
            "batch_size": 128, # 가상 데이터는 크기가 작으므로 배치 사이즈 늘림
            "lr": 0.001,
            "epochs": 50,
            "hidden_size": 256,
            "decoder": "ctc_beam_search"
        }
    )
    config = wandb.config

    # 이미지 전처리 정의 (가상 데이터에 맞게 조정)
    transform = transforms.Compose([
        transforms.Resize((32, 128)), # CRNN 입력 크기에 맞춤
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # 학습/검증 데이터셋 로딩 (가상 데이터 경로 사용)
    train_dataset = OCRDataset(
        image_dir=str(ocr_dir / 'data/virtual_data/images'),
        label_dir=str(ocr_dir / 'data/virtual_data/labels'),
        transform=transform
    )
    val_dataset = OCRDataset(
        image_dir=str(ocr_dir / 'data/virtual_val_data/images'),
        label_dir=str(ocr_dir / 'data/virtual_val_data/labels'),
        transform=transform
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True, # 샘플러 대신 사용
        num_workers=4,
        pin_memory=True,
        collate_fn=custom_collate_fn
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size, # 검증도 배치 처리
        shuffle=False,
        num_workers=2,
        collate_fn=custom_collate_fn
    )

    # 인코더 및 모델, 손실함수, 옵티마이저 구성
    encoder = LabelEncoder(charset=VIRTUAL_CHARSET)
    num_classes = encoder.num_classes()
    model = CRNN(num_classes=num_classes, hidden_size=config.hidden_size).to(device)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = optim.Adam(model.parameters(), lr=config.lr)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5)

    best_acc = 0.0
    log = []
    patience = 15
    counter = 0

    # 학습 루프
    try:
        for epoch in range(config.epochs):
            model.train()
            running_loss = 0.0
            for batch_idx, (images, texts) in enumerate(train_loader):
                images = images.to(device)

                targets = [torch.tensor(encoder.encode(t), dtype=torch.long) for t in texts]
                target_lengths = torch.tensor([len(t) for t in targets], dtype=torch.long)
                targets_concat = torch.cat(targets).to(device)

                batch_size = images.size(0)
                output = model(images)
                input_lengths = torch.full(size=(batch_size,), fill_value=output.size(0), dtype=torch.long)

                loss = criterion(output.log_softmax(2), targets_concat, input_lengths, target_lengths)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5) # Gradient Clipping
                optimizer.step()

                running_loss += loss.item()

            train_loss = running_loss / len(train_loader)
            val_acc, val_loss = evaluate(model, val_loader, encoder, criterion, device)
            scheduler.step(val_loss)

            # 샘플 예측
            pred_samples = []
            model.eval()
            with torch.no_grad():
                sample_indices = random.sample(range(len(val_dataset)), min(5, len(val_dataset)))
                for idx in sample_indices:
                    sample_img, sample_text = val_dataset[idx]
                    sample_img = sample_img.unsqueeze(0).to(device)
                    pred = model(sample_img)
                    pred_beam = decode_beam_search(pred, encoder, beam_width=5)[0]
                    acc = char_accuracy(sample_text, pred_beam)
                    pred_samples.append((sample_text, pred_beam, round(acc, 2)))

            print(f"[Epoch {epoch+1}/{config.epochs}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
            for gt, pred, acc in pred_samples:
                print(f"  [Sample] GT: '{gt}' -> Pred: '{pred}' (Acc: {acc}%)")

            # 모델 저장
            if val_acc > best_acc:
                best_acc = val_acc
                counter = 0
                torch.save(model.state_dict(), str(best_path))
                print(f"Best model updated! Saved to: {best_path} (Val Acc: {best_acc:.2f}%)")
            else:
                counter += 1
                print(f"No improvement for {counter}/{patience} epochs.")
                if counter >= patience:
                    print("Early stopping triggered!")
                    break

            # wandb 로깅
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "lr": optimizer.param_groups[0]["lr"]
            })
            log.append((epoch + 1, train_loss, val_loss, val_acc))

    except KeyboardInterrupt:
        print("\nTraining interrupted! Saving last state...")
        torch.save(model.state_dict(), ocr_dir / 'checkpoints/last_interrupted_v1.5.1.pth')
        print("Last state saved.")

    # CSV로 학습 로그 저장
    wandb.finish()
    log_dir = ocr_dir / "train_logs"
    log_dir.mkdir(exist_ok=True)
    df = pd.DataFrame(log, columns=['epoch', 'train_loss', 'val_loss', 'val_acc'])
    df.to_csv(log_dir / 'train_log_v1.5.1.csv', index=False)
    print(f"Training log saved: {log_dir / 'train_log_v1.5.1.csv'}")
