import torch
from torchvision import transforms
from PIL import Image
import yaml
import pandas as pd
import os
import torch.nn.functional as F
import joblib
from model import get_model

with open("config.yaml", 'r') as f:
    config = yaml.safe_load(f)

# 设置路径
metadata_dir = config['train_csv']
root_dir = config['train_images_dir']
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载数据和模型
df = pd.read_csv(metadata_dir)
query_df = df[df['split'] == 'query'].reset_index(drop=True)

encoder = joblib.load(config['label_encoder_path'])
model = get_model(config['model_name'], len(encoder.classes_))
model.load_state_dict(torch.load(config['model_output_path'], map_location=device))
model.to(device)
model.eval()

# 变换
transform = transforms.Compose([
    transforms.Resize((config['image_size'], config['image_size'])),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 预测
threshold = 0.5
results = []
for i, row in query_df.iterrows():
    img_path = os.path.join(root_dir, row['path'])
    img = Image.open(img_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)
        probs = F.softmax(outputs, dim=1)
        max_prob, pred_class = torch.max(probs, 1)

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
