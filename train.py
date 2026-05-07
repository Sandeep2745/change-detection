import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import segmentation_models_pytorch as smp
from dataset import ChangeDetectionDataset, get_transforms
from torch.utils.data import DataLoader
import yaml

# Load config
with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Datasets
train_transform, val_transform = get_transforms(cfg["image_size"])
train_dataset = ChangeDetectionDataset(cfg["train_pre"], cfg["train_post"], cfg["train_target"], train_transform)
val_dataset = ChangeDetectionDataset(cfg["val_pre"], cfg["val_post"], cfg["val_target"], val_transform)
train_loader = DataLoader(train_dataset, batch_size=cfg["batch_size"], shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=cfg["batch_size"], shuffle=False, num_workers=2)

# Model
model = smp.Unet(encoder_name=cfg["encoder"], encoder_weights="imagenet", in_channels=4, classes=2).to(device)

# Loss
class CombinedLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.dice = smp.losses.DiceLoss(mode="multiclass")
        self.ce = nn.CrossEntropyLoss(weight=torch.tensor([0.2, 0.8]).to(device))
    def forward(self, pred, target):
        return self.dice(pred, target) + self.ce(pred, target)

criterion = CombinedLoss()
optimizer = optim.Adam(model.parameters(), lr=cfg["learning_rate"])
scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=3, factor=0.5)

best_val_loss = float("inf")
for epoch in range(cfg["epochs"]):
    model.train()
    train_loss = 0
    for images, masks in train_loader:
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_loader)

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            val_loss += loss.item()
    val_loss /= len(val_loader)
    scheduler.step(val_loss)

    print(f"Epoch {epoch+1}/{cfg['epochs']} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), cfg["checkpoint_path"])
        print("  Best model saved!")

print("Training Complete!")
