import pandas as pd
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
import os

class CLEFDataset(Dataset):
    def __init__(self, df, root_dir,transform=None):
        self.df = df.reset_index(drop=True)
        self.root_dir = root_dir
        if transform:
            self.transform=transform
        self.transform = transforms.Compose([
                            transforms.Resize((224, 224)),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406],
                                                  [0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.loc[idx, 'path']
        # print(img_path)
        img_path = os.path.join(self.root_dir, img_path)
        label = self.df.loc[idx, 'label']
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label