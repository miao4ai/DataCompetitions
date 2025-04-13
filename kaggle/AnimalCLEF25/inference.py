import torch
from models.model import CLEFModel
from utils.utils import load_image
import yaml
import os

def predict():
    with open("config/config.yaml") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CLEFModel(model_name=cfg["model"], num_classes=cfg["num_classes"])
    model.load_state_dict(torch.load(cfg["model_path"], map_location=device))
    model.to(device)
    model.eval()

    image_path = cfg["infer_image"]
    image = load_image(image_path).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        pred = torch.argmax(output, dim=1).item()
    print(f"Prediction: {pred}")

if __name__ == '__main__':
    predict()
