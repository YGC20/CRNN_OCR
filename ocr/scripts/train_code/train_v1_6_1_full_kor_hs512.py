


# train_v1_6_0_full_kor_hs512.py - 전체 한글 가상 데이터셋, hidden_size=512, OneCycleLR 적용

import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import wandb
import random
import numpy as np
from torch.utils.data import DataLoader
from collections import Counter
from torchvision import transforms
from pathlib import Path

from utils.dataset import OCRDataset
from utils.label_encoder import LabelEncoder
from model.crnn import CRNN

# --- 전체 가상 문자셋 정의 ---
# 1. 완성형 한글
KOREAN_CHARS = '가강거경계고관광구금기김나남너노누다대더도동두등라러로루리마머명모무문미바버배뱌버보부북사산서소수아악안양어연영오용우울원육이인자작저전조주중지차천처초추충카커코쿠타터토투파평퍼포푸하허호홀후후히'
# 2. 한글 자모 (자음, 모음)
JAMO_CHARS = "ㄱㄴㄷㄹㅁㅂㅅㅇㅈㅊㅋㅌㅍㅎㅏㅑㅓㅕㅗㅛㅜㅠㅡㅣㄲㄸㅃㅆㅉㅐㅒㅔㅖㅘㅙㅚㅝㅞㅟㅢ"
# 3. 영문 및 숫자
ENGLISH_CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
NUMBER_CHARS = "0123456789"
# 4. 최종 문자셋 결합
VIRTUAL_CHARSET = KOREAN_CHARS + JAMO_CHARS + ENGLISH_CHARS + NUMBER_CHARS
# --- 문자셋 정의 끝 ---

# 모델 저장 경로 수정
ocr_dir = Path(__file__).resolve().parent
project_root = ocr_dir.parent
bestmodel_dir = project_root / "bestModel"
bestmodel_dir.mkdir(exist_ok=True)
best_path = bestmodel_dir / "ocrBestModel_16_full_kor_hs512.pth" # 새 모델 경로

def char_accuracy(gt, pred):
    if not gt or not pred:
        return 0.0
    match = sum(g == p for g, p in zip(gt, pred))
    return match / max(len(gt), len(pred)) * 100

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

def evaluate(model, val_loader, encoder, criterion, device):
    model.eval()
    total = 0
    correct = 0
    val_loss = 0.0
    with torch.no_grad():
        for images, texts in val_loader:
            images = images.to(device)
            outputs = model(images)
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

def custom_collate_fn(batch):
    images, texts = zip(*batch)
    images = torch.stack(images, dim=0)
    return images, list(texts)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    wandb.init(
        project="crnn-ocr-full-kor-hs512", # 새 프로젝트 이름
        name="train_v1.6.1_full_kor_hs512",
        dir=str(ocr_dir / "wandb_logs"),
        config={
            "batch_size": 128,
            "max_lr": 0.001, # OneCycleLR을 위한 max_lr
            "epochs": 100, # 에포크 늘림
            "hidden_size": 512, # hidden_size 증가
            "decoder": "ctc_beam_search"
        }
    )
    config = wandb.config

    transform = transforms.Compose([
        transforms.Resize((32, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # 전체 한글 데이터셋 경로 사용
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
        train_dataset, batch_size=config.batch_size, shuffle=True, 
        num_workers=4, pin_memory=True, collate_fn=custom_collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.batch_size, shuffle=False,
        num_workers=2, collate_fn=custom_collate_fn
    )

    encoder = LabelEncoder(charset=VIRTUAL_CHARSET)
    num_classes = encoder.num_classes()
    model = CRNN(num_classes=num_classes, hidden_size=config.hidden_size).to(device)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = optim.Adam(model.parameters(), lr=config.max_lr)
    
    # OneCycleLR 스케줄러 설정
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config.max_lr,
        epochs=config.epochs,
        steps_per_epoch=len(train_loader)
    )

    best_acc = 0.0
    log = []
    patience = 20 # 조기 종료 조건 완화
    counter = 0

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
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
                optimizer.step()
                scheduler.step() # 매 스텝마다 스케줄러 업데이트
                running_loss += loss.item()

            train_loss = running_loss / len(train_loader)
            val_acc, val_loss = evaluate(model, val_loader, encoder, criterion, device)

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

            wandb.log({
                "epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss,
                "val_acc": val_acc, "lr": optimizer.param_groups[0]["lr"]
            })
            log.append((epoch + 1, train_loss, val_loss, val_acc))

    except KeyboardInterrupt:
        print("\nTraining interrupted! Saving last state...")
        torch.save(model.state_dict(), ocr_dir / 'checkpoints/last_interrupted_v1.6.0.pth')
        print("Last state saved.")

    wandb.finish()
    log_dir = ocr_dir / "train_logs"
    log_dir.mkdir(exist_ok=True)
    df = pd.DataFrame(log, columns=['epoch', 'train_loss', 'val_loss', 'val_acc'])
    df.to_csv(log_dir / 'train_log_v1.6.0.csv', index=False)
    print(f"Training log saved: {log_dir / 'train_log_v1.6.0.csv'}")

