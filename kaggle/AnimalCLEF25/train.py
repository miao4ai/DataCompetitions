import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import yaml
import pandas as pd
import joblib
import os
from sklearn.preprocessing import LabelEncoder
from model import get_model
from utils import CLEFDataset
from tqdm import tqdm

def main():
    with open("config.yaml", 'r') as f:
        config = yaml.safe_load(f)

    os.makedirs(config['model_registry_dir'], exist_ok=True)

    df = pd.read_csv(config['train_csv'])
    database_df = df[df["split"] == "database"].dropna(subset=['identity']).reset_index(drop=True)

    encoder = LabelEncoder()
    database_df['label'] = encoder.fit_transform(database_df['identity'])
    joblib.dump(encoder, config['label_encoder_path'])
    num_classes = len(encoder.classes_)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    dataset = CLEFDataset(database_df, config['train_images_dir'], transform=transform)
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)

    # Use ViT from scratch
    model = get_model('vit_base_patch16_224', num_classes)
    model = torch.hub.load('rwightman/pytorch-image-models', 'vit_base_patch16_224', pretrained=False)
    model.head = nn.Linear(model.head.in_features, num_classes)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    best_loss = float('inf')
    patience_counter = 0

    for epoch in range(50):
        model.train()
        running_loss = 0.0
        pbar = tqdm(dataloader, desc=f"ViT Scratch Epoch [{epoch+1}/50]")
        for imgs, labels in pbar:
            imgs, labels = imgs.to(device), labels.to(device).long()
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            avg_loss = running_loss / (pbar.n + 1)
            pbar.set_postfix(loss=avg_loss)

        epoch_loss = running_loss / len(dataloader)
        print(f"✅ Epoch {epoch+1} finished. Average Loss: {epoch_loss:.4f}")

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            patience_counter = 0
            best_model_path = os.path.join(config['model_registry_dir'], "vit_scratch_best.pt")
            torch.save(model.state_dict(), best_model_path)
            print("🔃 Saved best model to:", best_model_path)
        else:
            patience_counter += 1
            print(f"⏸ No improvement. Patience: {patience_counter}/10")
            if patience_counter >= 10:
                print("⛔ Early stopping triggered.")
                break

if __name__ == '__main__':
    main()