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

def train_one_model(model_name, config, database_df, encoder):
    model_input_sizes = {
        'swinv2_base_window12to16_192to256_22kft1k': 256,
        'vit_base_patch16_224': 224,
        'vit_base_patch32_224': 224,
        'convnext_base': 224,
        'coatnet_0': 224
    }
    image_size = model_input_sizes.get(model_name, config['image_size'])

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    dataset = CLEFDataset(database_df, config['train_images_dir'], transform=transform)
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)

    num_classes = len(encoder.classes_)
    model = get_model(model_name, num_classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=float(config['learning_rate']))

    best_loss = float('inf')
    patience_counter = 0
    for epoch in range(config['epochs']):
        model.train()
        running_loss = 0.0
        pbar = tqdm(dataloader, desc=f"{model_name} Epoch [{epoch+1}/{config['epochs']}]")
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
        print(f"✅ {model_name} Epoch {epoch+1} finished. Average Loss: {epoch_loss:.4f}")

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            patience_counter = 0
            best_model_path = os.path.join(config['model_registry_dir'], f"{model_name}_best.pt")
            torch.save(model.state_dict(), best_model_path)
            print("🔃 Saved best model to:", best_model_path)
        else:
            patience_counter += 1
            print(f"⏸ No improvement. Patience: {patience_counter}/{config['early_stopping_patience']}")
            if patience_counter >= config['early_stopping_patience']:
                print("⛔ Early stopping triggered.")
                break

def main():
    with open("config.yaml", 'r') as f:
        config = yaml.safe_load(f)

    os.makedirs(config['model_registry_dir'], exist_ok=True)

    df = pd.read_csv(config['train_csv'])
    database_df = df[df["split"] == "database"].dropna(subset=['identity']).reset_index(drop=True)

    encoder = LabelEncoder()
    database_df['label'] = encoder.fit_transform(database_df['identity'])
    joblib.dump(encoder, config['label_encoder_path'])

    for model_name in config['ensemble_model_names']:
        train_one_model(model_name, config, database_df, encoder)

if __name__ == '__main__':
    main()