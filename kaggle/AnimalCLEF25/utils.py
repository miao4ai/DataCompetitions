import torch
from torchvision import transforms
from PIL import Image

def get_transforms():
    return transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])

def load_image(path):
    image = Image.open(path).convert("RGB")
    transform = get_transforms()
    return transform(image)
