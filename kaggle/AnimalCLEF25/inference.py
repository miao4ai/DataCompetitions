import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import yaml
import pandas as pd
import joblib
import os
from tqdm import tqdm

def main():
    with open("config.yaml", 'r') as f:
        config = yaml.safe_load(f)

    metadata_path = config['train_csv']
    root_dir = config['train_images_dir']
    model_path = os.path.join(config['model_registry_dir'], "vit_scratch_best.pt")
    encoder = joblib.load(config['label_encoder_path'])
    num_classes = len(encoder.classes_)

    df = pd.read_csv(metadata_path)
    query_df = df[df["split"] == "query"].reset_index(drop=True)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    model = torch.hub.load('rwightman/pytorch-image-models', 'vit_base_patch16_224', pretrained=False)
    model.head = torch.nn.Linear(model.head.in_features, num_classes)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    results = []
    threshold = 0.5

    for _, row in tqdm(query_df.iterrows(), total=len(query_df), desc="Predicting"):
        img_path = os.path.join(root_dir, row['path'])
        img = Image.open(img_path).convert("RGB")
        img_tensor = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(img_tensor)
            prob = F.softmax(output, dim=1)
            max_prob, pred = torch.max(prob, dim=1)

        if max_prob.item() >= threshold:
            identity = encoder.inverse_transform([pred.item()])[0]
        else:
            identity = "new_individual"

        results.append({"image_id": row["image_id"], "identity": identity})

    submission = pd.DataFrame(results)
    submission.to_csv("submission_vit_scratch.csv", index=False)
    print("📤 Saved prediction to submission_vit_scratch.csv")

if __name__ == '__main__':
    main()
