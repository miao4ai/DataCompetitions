# inference.py
import torch
from torchvision import transforms
from PIL import Image
import yaml
import pandas as pd
import os
import torch.nn.functional as F
import joblib
from model import get_model

def main():
    with open("config.yaml", 'r') as f:
        config = yaml.safe_load(f)

    metadata_dir = config['train_csv']
    root_dir = config['train_images_dir']
    model_dir = config['model_registry_dir']
    model_name = config['model_name']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = pd.read_csv(metadata_dir)
    query_df = df[df['split'] == 'query'].reset_index(drop=True)
    encoder = joblib.load(config['label_encoder_path'])
    num_classes = len(encoder.classes_)

    model = get_model(model_name, num_classes)
    model_path = os.path.join(model_dir, f"{model_name}_best.pt")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((config['image_size'], config['image_size'])),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    threshold = 0.5
    results = []

    for _, row in query_df.iterrows():
        img_path = os.path.join(root_dir, row['path'])
        img = Image.open(img_path).convert("RGB")
        img_tensor = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(img_tensor)
            prob = F.softmax(output, dim=1)
            max_prob, pred_class = torch.max(prob, dim=1)

        if max_prob.item() >= threshold:
            identity = encoder.inverse_transform([pred_class.item()])[0]
        else:
            identity = "new_individual"

        results.append({
            "image_id": row["image_id"],
            "identity": identity
        })

    submission_df = pd.DataFrame(results)
    submission_df.to_csv("submission.csv", index=False)
    print("✅ Saved predictions to submission.csv")

if __name__ == '__main__':
    main()
