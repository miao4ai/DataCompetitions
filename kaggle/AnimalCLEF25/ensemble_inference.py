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

    model_input_sizes = {
        'swinv2_base_window12to16_192to256_22kft1k': 256,
        'vit_base_patch16_224': 224,
        'vit_base_patch32_224': 224,
        'convnext_base': 224,
        'vit_base_patch16_224_in21k': 224,
        'beit_base_patch16_224': 224
    }

    all_model_probs = []

    for model_name in model_names:
        image_size = model_input_sizes.get(model_name, config['image_size'])
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        model = get_model(model_name, num_classes)
        model_path = os.path.join(model_dir, f"{model_name}_best.pt")
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()

        def enable_dropout(m):
            if isinstance(m, torch.nn.Dropout):
                m.train()
        model.apply(enable_dropout)

        model_probs = []
        for _, row in query_df.iterrows():
            img_path = os.path.join(root_dir, row['path'])
            img = Image.open(img_path).convert("RGB")

            # TTA
            variations = [
                transform(img),
                transform(transforms.functional.hflip(img)),
                transform(transforms.functional.adjust_brightness(img, 1.2)),
                transform(transforms.functional.adjust_contrast(img, 1.2))
            ]
            imgs_tta = torch.stack(variations).to(device)

            with torch.no_grad():
                N = 5
                probs_mc = []
                for _ in range(N):
                    outputs = model(imgs_tta)
                    probs_mc.append(F.softmax(outputs, dim=1).mean(dim=0).cpu().numpy())
                prob = np.mean(probs_mc, axis=0)
                model_probs.append(prob)

        all_model_probs.append(np.stack(model_probs))

    avg_probs = np.mean(np.stack(all_model_probs), axis=0)
    max_probs = np.max(avg_probs, axis=1)
    preds = np.argmax(avg_probs, axis=1)

    # 自适应 threshold
    mean_conf = np.mean(max_probs)
    std_conf = np.std(max_probs)
    adaptive_threshold = mean_conf - 0.5 * std_conf
    print(f"🧠 Adaptive Threshold: {adaptive_threshold:.4f}")

    pred_labels = []
    for prob, p in zip(max_probs, preds):
        if prob >= adaptive_threshold:
            label = encoder.inverse_transform([p])[0]
        else:
            label = "new_individual"
        pred_labels.append(label)

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