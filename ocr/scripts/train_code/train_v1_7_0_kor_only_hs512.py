
import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import wandb
import random
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from pathlib import Path
from PIL import Image

from utils.label_encoder import LabelEncoder
from model.crnn import CRNN

# --- 한글 전용 문자셋 정의 (영문 제외) --- 
# 1. 완성형 한글 + 자모
HANGUL_CHARS = [chr(0xAC00 + i) for i in range(11172)]
initial_consonants = [chr(i) for i in range(0x1100, 0x1113)]
vowels = [chr(i) for i in range(0x1161, 0x1176)]
final_consonants = [chr(i) for i in range(0x11A8, 0x11C3)]
JAMO_CHARS = initial_consonants + vowels + final_consonants
# 2. 숫자
NUMBER_CHARS = "0123456789"
# 3. 특수문자
SPECIAL_CHARS = ".,!@#$%^&"
# 4. 최종 문자셋 결합
KOR_CHARSET = "".join(sorted(list(set(HANGUL_CHARS + JAMO_CHARS + list(NUMBER_CHARS) + list(SPECIAL_CHARS)))))
# --- 문자셋 정의 끝 ---

# --- 경로 설정 ---
ocr_dir = Path(__file__).resolve().parent
project_root = ocr_dir.parent
bestmodel_dir = project_root / "bestModel"
bestmodel_dir.mkdir(exist_ok=True)
best_path = bestmodel_dir / "ocrBestModel_170_kor_only_hs512.pth" # 새 모델 경로

# --- 새로운 데이터셋 클래스 정의 ---
class SimpleOCRDataset(Dataset):
    """labels.txt를 직접 읽어 이미지를 로드하는 간단한 데이터셋 클래스"""
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.samples = []
        
        label_file_path = os.path.join(data_dir, "labels.txt")
        with open(label_file_path, "r", encoding="utf-8") as f:
            for line in f.readlines():
                filename, label = line.strip().split("\t")
                self.samples.append((filename, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        filename, label = self.samples[idx]
        image_path = os.path.join(self.data_dir, filename)
        
        try:
            image = Image.open(image_path).convert('L')
            if self.transform:
                image = self.transform(image)
            return image, label
        except FileNotFoundError:
            print(f"Warning: Image file not found {image_path}. Skipping.")
            # 다음 샘플을 시도
            return self.__getitem__((idx + 1) % len(self))

# --- 유틸리티 함수 (기존과 동일) ---
def decode_beam_search(logits, encoder, beam_width=5):
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
            decoded_texts = decode_beam_search(outputs, encoder)
            for pred_text, gt_text in zip(decoded_texts, texts):
                if gt_text == pred_text:
                    correct += 1
                total += 1
    acc = (correct / total * 100) if total > 0 else 0
    avg_loss = (val_loss / len(val_loader)) if len(val_loader) > 0 else 0
    return acc, avg_loss

def custom_collate_fn(batch):
    batch = [item for item in batch if item is not None] # Skip None items from missing files
    if not batch:
        return torch.empty(0), []
    images, texts = zip(*batch)
    images = torch.stack(images, dim=0)
    return images, list(texts)

# --- 메인 학습 로직 ---
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    wandb.init(
        project="crnn-ocr-kor-only-hs512", # 새 프로젝트 이름
        name="train_v1.7.0_kor_only_hs512",
        dir=str(ocr_dir / "wandb_logs"),
        config={
            "batch_size": 128,
            "max_lr": 0.001,
            "epochs": 100,
            "hidden_size": 512,
            "charset": "Korean_Only"
        }
    )
    config = wandb.config

    transform = transforms.Compose([
        transforms.Resize((32, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # SimpleOCRDataset 사용
    train_dataset = SimpleOCRDataset(
        data_dir=str(ocr_dir / 'data/virtual_data_kor'),
        transform=transform
    )
    val_dataset = SimpleOCRDataset(
        data_dir=str(ocr_dir / 'data/virtual_val_data_kor'),
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

    encoder = LabelEncoder(charset=KOR_CHARSET)
    num_classes = encoder.num_classes()
    model = CRNN(num_classes=num_classes, hidden_size=config.hidden_size).to(device)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = optim.Adam(model.parameters(), lr=config.max_lr)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=config.max_lr, epochs=config.epochs, steps_per_epoch=len(train_loader)
    )

    best_acc = 0.0
    patience = 20
    counter = 0

    print("--- Starting Training (Korean-Only Dataset) ---")
    try:
        for epoch in range(config.epochs):
            model.train()
            running_loss = 0.0
            for images, texts in train_loader:
                if not images.numel(): continue # Skip empty batches
                images = images.to(device)
                targets = [torch.tensor(encoder.encode(t), dtype=torch.long) for t in texts]
                target_lengths = torch.tensor([len(t) for t in targets], dtype=torch.long)
                targets_concat = torch.cat(targets).to(device)
                output = model(images)
                input_lengths = torch.full(size=(images.size(0),), fill_value=output.size(0), dtype=torch.long)
                loss = criterion(output.log_softmax(2), targets_concat, input_lengths, target_lengths)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
                optimizer.step()
                scheduler.step()
                running_loss += loss.item()

            train_loss = running_loss / len(train_loader) if len(train_loader) > 0 else 0.0
            val_acc, val_loss = evaluate(model, val_loader, encoder, criterion, device)

            print(f"[Epoch {epoch+1}/{config.epochs}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

            if val_acc > best_acc:
                best_acc = val_acc
                counter = 0
                torch.save(model.state_dict(), str(best_path))
                print(f"Best model updated! Saved to: {best_path} (Val Acc: {best_acc:.2f}%)")
            else:
                counter += 1
                if counter >= patience:
                    print(f"No improvement for {counter}/{patience} epochs. Early stopping triggered!")
                    break

            wandb.log({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss, "val_acc": val_acc, "lr": optimizer.param_groups[0]['lr']})

    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    finally:
        wandb.finish()
        print(f"Training finished. Best validation accuracy: {best_acc:.2f}%")
        print(f"Best model saved at: {best_path}")
