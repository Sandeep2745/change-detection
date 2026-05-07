import torch
import numpy as np
import argparse
import os
from PIL import Image
import segmentation_models_pytorch as smp
from dataset import ChangeDetectionDataset, get_transforms
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", required=True)
parser.add_argument("--weights", required=True)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=4, classes=2).to(device)
model.load_state_dict(torch.load(args.weights, map_location=device))
model.eval()

_, val_transform = get_transforms(256)
dataset = ChangeDetectionDataset(
    os.path.join(args.data_path, "pre-event"),
    os.path.join(args.data_path, "post-event"),
    os.path.join(args.data_path, "target"),
    val_transform
)
loader = DataLoader(dataset, batch_size=4, shuffle=False)

all_preds, all_masks = [], []
with torch.no_grad():
    for images, masks in loader:
        images = images.to(device)
        outputs = model(images)
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy().flatten())
        all_masks.extend(masks.numpy().flatten())

all_preds = np.array(all_preds)
all_masks = np.array(all_masks)

tp = np.sum((all_preds == 1) & (all_masks == 1))
fp = np.sum((all_preds == 1) & (all_masks == 0))
fn = np.sum((all_preds == 0) & (all_masks == 1))

precision = tp / (tp + fp + 1e-8)
recall = tp / (tp + fn + 1e-8)
f1 = 2 * precision * recall / (precision + recall + 1e-8)
iou = tp / (tp + fp + fn + 1e-8)

print(f"IoU:       {iou:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")
print(f"Confusion Matrix:")
print(confusion_matrix(all_masks, all_preds))
