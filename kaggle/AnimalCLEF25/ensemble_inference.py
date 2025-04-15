import torch
from torchvision import transforms
from PIL import Image
import yaml
import pandas as pd
import os
import torch.nn.functional as F
import joblib
import numpy as np
from datetime import datetime
from model import get_model


def ensemble_predict(model_names, config):
    metadata_path = config['train_csv']
    root_dir = config['train_images_dir']
    model_dir = config['model_registry_dir']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = pd.read_csv(metadata_path)
    query_df = df[df['split'] == 'query'].reset_index(drop=True)
    encoder = joblib.load(config['label_encoder_path'])
    num_classes = len(encoder.classes_)

    base_transform = transforms.Compose([
        transforms.Resize((config['image_size'], config['image_size'])),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    def tta_transforms(img):
        variations = [
            base_transform(img),
            base_transform(transforms.functional.hflip(img)),
            base_transform(transforms.functional.adjust_brightness(img, 1.2)),
            base_transform(transforms.functional.adjust_contrast(img, 1.2))
        ]
        return torch.stack(variations)

    threshold = 0.5
    all_model_probs = []

    for model_name in model_names:
        model = get_model(model_name, num_classes)
        model_path = os.path.join(model_dir, f"{model_name}_best.pt")
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()

        model_probs = []
        for _, row in query_df.iterrows():
            img_path = os.path.join(root_dir, row['path'])
            img = Image.open(img_path).convert("RGB")
            imgs_tta = tta_transforms(img).to(device)

            with torch.no_grad():
                output = model(imgs_tta)
                prob = F.softmax(output, dim=1).mean(dim=0).cpu().numpy()
                model_probs.append(prob)

        all_model_probs.append(np.stack(model_probs))

    avg_probs = np.mean(np.stack(all_model_probs), axis=0)
    preds = np.argmax(avg_probs, axis=1)
    max_probs = np.max(avg_probs, axis=1)

    pred_labels = []
    for p, prob in zip(preds, max_probs):
        if prob >= threshold:
            pred_labels.append(encoder.inverse_transform([p])[0])
        else:
            pred_labels.append("new_individual")

    submission = pd.DataFrame({
        "image_id": query_df["image_id"],
        "identity": pred_labels
    })

    os.makedirs("submission", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    submission_path = f"submission/submission_{timestamp}.csv"
    submission.to_csv(submission_path, index=False)
    print("📤 Saved predictions to:", submission_path)

if __name__ == '__main__':
    with open("config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    ensemble_predict(config['ensemble_model_names'], config)

