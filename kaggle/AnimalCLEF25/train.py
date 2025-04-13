import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import yaml
from model import get_model
from utils import CLEFDataset

with open("config.yaml", 'r') as f:
    config = yaml.safe_load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((config['image_size'], config['image_size'])),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

dataset = CLEFDataset(config['train_csv'], config['train_images_dir'], transform=transform)
dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)

model = get_model(config['model_name'], config['num_classes']).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'])

best_acc = 0.0
for epoch in range(config['epochs']):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    acc = correct / total
    print(f"Epoch {epoch+1}, Loss: {running_loss/total:.4f}, Accuracy: {acc:.4f}")
    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), config['model_output_path'])