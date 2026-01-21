import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pandas as pd
from PIL import Image
import os

class ChestXrayDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None, mode='dense'):
        """
        Args:
            mode (str): 'dense' (Multi-label) hoặc 'alex' (Single-label)
        """
        self.img_dir = img_dir
        self.transform = transform
        self.mode = mode
        
        df = pd.read_csv(csv_file)
        self.classes = sorted(df['class_name'].unique())
        self.c2i = {c: i for i, c in enumerate(self.classes)}
        
        if mode == 'dense':
            # Gom nhóm 1 ảnh -> nhiều bệnh
            self.data = df.groupby('image_id')['class_name'].apply(list).reset_index()
        else:
            self.data = df

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        if self.mode == 'dense':
            row = self.data.iloc[idx]
            img_id = row['image_id']
            labels = row['class_name']
        else:
            row = self.data.iloc[idx]
            img_id = row['image_id']
            label_name = row['class_name']

        exts = ['.jpg', '.png', '.jpeg']
        for ext in exts:
            path = os.path.join(self.img_dir, str(img_id) + ext)
            if os.path.exists(img_path):
                img_path = path
                break
        
        try:
            image = Image.open(img_path).convert("RGB")
        except:
            image = Image.new('RGB', (227, 227)) # Ảnh đen placeholder

        if self.transform: image = self.transform(image)

        # Xử lý Label (Target)
        if self.mode == 'dense':
            target = torch.zeros(len(self.classes))
            for cls in labels:
                if cls in self.c2i: target[self.c2i[cls]] = 1.0
        else:
            # AlexNet cần label dạng số nguyên (long)
            target = torch.tensor(self.c2i[label_name], dtype=torch.long)

        return image, target

# Helper function để lấy nhanh Loader
def get_loaders(data_root, model_type, batch_size=32):
    csv_train = os.path.join(data_root, f"train_{model_type}.csv")
    csv_val = os.path.join(data_root, f"val_{model_type}.csv")
    img_dir = os.path.join(data_root, "train_images")

    tf = transforms.Compose([
        transforms.Resize((227, 227)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_ds = ChestXrayDataset(csv_train, img_dir, tf, mode=model_type)
    val_ds = ChestXrayDataset(csv_val, img_dir, tf, mode=model_type)

    return (DataLoader(train_ds, batch_size, shuffle=True), 
            DataLoader(val_ds, batch_size, shuffle=False), 
            train_ds.classes)