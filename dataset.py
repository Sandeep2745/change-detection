import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

class ChangeDetectionDataset(Dataset):
    def __init__(self, pre_dir, post_dir, target_dir, transform=None):
        self.pre_dir = pre_dir
        self.post_dir = post_dir
        self.target_dir = target_dir
        self.files = sorted(os.listdir(pre_dir))
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        pre = np.array(Image.open(os.path.join(self.pre_dir, fname)).convert("RGB"))
        post = np.array(Image.open(os.path.join(self.post_dir, fname)).convert("L"))
        mask = np.array(Image.open(os.path.join(self.target_dir, fname)))
        mask = np.where(mask >= 2, 1, 0).astype(np.uint8)
        post = np.expand_dims(post, axis=-1)
        image = np.concatenate([pre, post], axis=-1)
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]
        return image.float(), mask.long()

def get_transforms(image_size):
    train_transform = A.Compose([
        A.Resize(image_size, image_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Normalize(mean=[0.485, 0.456, 0.406, 0.5], std=[0.229, 0.224, 0.225, 0.5]),
        ToTensorV2()
    ])
    val_transform = A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=[0.485, 0.456, 0.406, 0.5], std=[0.229, 0.224, 0.225, 0.5]),
        ToTensorV2()
    ])
    return train_transform, val_transform
