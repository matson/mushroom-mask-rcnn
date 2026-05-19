
# -------- INSTALLATION --------

'''
pip install torch torchvision torchaudio
pip install matplotlib
pip install pillow
pip install pycocotools
pip install tqdm
pip install gradio
pip install transformers
pip install huggingface_hubs
pip install albumentations
'''

# -------- IMPORTS --------
import torch
from torch.utils.data import Dataset, DataLoader
import json

import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2, MaskRCNN_ResNet50_FPN_V2_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from pycocotools.coco import COCO
import albumentations as A

import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils import evaluate_mAP
from utils import save_checkpoint

'''
Mask R-CNN v2 — stronger backbone with gradient clipping.
Categories:
  COCO category_id 0 (Mushrooms generic) → model label 1
  COCO category_id 1 (BB)               → model label 2
  COCO category_id 2 (WB)               → model label 3
  background                             → model label 0
num_classes = 4
'''

# ------------------- DEVICE & MEMORY -------------------
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
torch.cuda.empty_cache()
torch.backends.cuda.max_split_size_mb = 64

DEBUG_PRINTS = False

# -------- AUGMENTATIONS --------
augmentations = A.Compose([
    A.Affine(
        scale=(0.9, 1.1),
        rotate=(-20, 20),
        translate_percent=(-0.1, 0.1),
        fit_output=True,
        p=0.6
    ),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.GaussianBlur(p=0.3),
    A.HueSaturationValue(p=0.4)
])


def mask_to_box(mask):
    """Convert a binary H x W mask to [x1, y1, x2, y2] bounding box."""
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any() or not cols.any():
        return None
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return [float(cmin), float(rmin), float(cmax + 1), float(rmax + 1)]


# -------- CUSTOM DATASET CLASS --------
class MushroomCOCODataset(Dataset):
    def __init__(self, images_dir, annotations_file, augmentations=None, resize=(640, 640)):
        self.images_dir = images_dir
        self.coco = COCO(annotations_file)
        self.augmentations = augmentations
        self.resize = resize

        self.img_ids = [
            img_id for img_id in self.coco.imgs.keys()
            if len(self.coco.getAnnIds(imgIds=img_id)) > 0
        ]

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.images_dir, img_info['file_name'])
        image = Image.open(img_path).convert("RGB")
        w_original, h_original = image.size

        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)

        masks = []
        labels = []

        for ann in anns:
            x, y, width, height = ann['bbox']
            if width <= 1 or height <= 1:
                continue

            mask = self.coco.annToMask(ann)
            masks.append(mask)
            labels.append(ann['category_id'] + 1)  # 0→1, 1→2, 2→3

        if len(masks) == 0:
            raise ValueError(f"Image {img_id} has no valid annotations.")

        # Resize image and masks
        w_new, h_new = self.resize
        image = image.resize((w_new, h_new), resample=Image.BILINEAR)
        resized_masks = []
        for mask in masks:
            mask_img = Image.fromarray(mask.astype(np.uint8))
            mask_img = mask_img.resize((w_new, h_new), resample=Image.NEAREST)
            resized_masks.append(np.array(mask_img))

        # Augmentations applied to image + masks together
        if self.augmentations:
            transformed = self.augmentations(
                image=np.array(image),
                masks=resized_masks
            )
            image = transformed['image']
            resized_masks = transformed['masks']

        # Derive bounding boxes from augmented masks
        boxes = []
        valid_masks = []
        valid_labels = []
        for mask, label in zip(resized_masks, labels):
            box = mask_to_box(mask)
            if box is None:
                continue
            x1, y1, x2, y2 = box
            if (x2 - x1) <= 1 or (y2 - y1) <= 1:
                continue
            boxes.append(box)
            valid_masks.append(mask)
            valid_labels.append(label)

        if len(boxes) == 0:
            raise ValueError(f"Image {img_id} has no valid masks after augmentation.")

        image = torch.as_tensor(np.array(image), dtype=torch.float32).permute(2, 0, 1) / 255.0
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(valid_labels, dtype=torch.int64)
        masks_tensor = torch.as_tensor(np.stack(valid_masks), dtype=torch.uint8)

        target = {
            "boxes": boxes,
            "labels": labels,
            "masks": masks_tensor,
            "image_id": torch.tensor([img_id])
        }

        return image, target


# -------- DATALOADING --------
best_val_loss = float('inf')
batch_size = 1
accum_steps = 1

def collate_fn(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)

train_dataset = MushroomCOCODataset(
    images_dir=r"C:\data\M18KV2\train\rgb",
    annotations_file=r"C:\data\M18KV2\train\annotations_coco.json",
    augmentations=augmentations,
    resize=(640, 640)
)

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=2,
    pin_memory=True
)

val_dataset = MushroomCOCODataset(
    images_dir=r"C:\data\M18KV2\valid\rgb",
    annotations_file=r"C:\data\M18KV2\valid\annotations_coco.json",
    augmentations=None,
    resize=(640, 640)
)

val_loader = DataLoader(
    val_dataset,
    batch_size=1,
    shuffle=False,
    collate_fn=collate_fn,
    num_workers=2,
    pin_memory=True
)

# -------- LOAD MODEL (Mask R-CNN v2) --------
num_classes = 4  # background + Mushrooms + BB + WB

weights = MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT
model = maskrcnn_resnet50_fpn_v2(
    weights=weights,
    rpn_post_nms_top_n_train=500,
    box_detections_per_img=220
)

# Replace box predictor head
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

# Replace mask predictor head
in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, 256, num_classes)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

# -------- OPTIMIZER --------
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.0005, momentum=0.9, weight_decay=0.0001)

# -------- RESUME FROM CHECKPOINT --------
checkpoint_path = "best_maskrcnn_v2_mushroom_FULL.pth"
start_epoch = 81
best_val_loss = float('inf')

if os.path.exists(checkpoint_path):
    print(f"--- Loading Checkpoint: {checkpoint_path} ---")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    best_val_loss = checkpoint.get('best_val_loss', float('inf'))
    print(f"Resuming from epoch {start_epoch} (best val loss: {best_val_loss:.4f})")
else:
    print("No checkpoint found — starting from scratch.")

# -------- SCHEDULER --------
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=20, T_mult=1, eta_min=5e-6
)

# -------- TRAINING LOOP --------
def main():

    accum_steps = 8

    print("entering training")
    global start_epoch, best_val_loss
    train_losses, val_losses = [], []

    for epoch in range(start_epoch, start_epoch + 20):

        model.train()
        epoch_loss = 0
        optimizer.zero_grad()
        loop = tqdm(train_loader, total=len(train_loader), desc=f"Epoch [{epoch}/{start_epoch + 19}]")

        for batch_idx, (images, targets) in enumerate(loop):

            def mem(label):
                if not DEBUG_PRINTS:
                    return
                torch.cuda.synchronize()
                alloc = torch.cuda.memory_allocated() / 1024**2
                peak  = torch.cuda.max_memory_allocated() / 1024**2
                print(f"  [Batch {batch_idx}] {label:<20} Alloc: {alloc:.1f} MB | Peak: {peak:.1f} MB")

            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            mem("after data→GPU")

            if DEBUG_PRINTS:
                print(f"  [Batch {batch_idx}] boxes in image: {[len(t['boxes']) for t in targets]}")

            loss_dict = model(images, targets)
            mem("after forward")

            batch_loss = sum(loss for loss in loss_dict.values())
            if DEBUG_PRINTS:
                print(f"  [Batch {batch_idx}] losses: { {k: f'{v.item():.4f}' for k, v in loss_dict.items()} }")

            (batch_loss / accum_steps).backward()
            mem("after backward")

            if (batch_idx + 1) % accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
                mem("after optimizer")

            epoch_loss += batch_loss.item()
            running_avg_loss = epoch_loss / (batch_idx + 1)

            loop.set_postfix(batch_loss=f"{batch_loss.item():.4f}", avg_loss=f"{running_avg_loss:.4f}")

            del images, targets, loss_dict, batch_loss
            torch.cuda.empty_cache()
            mem("after cleanup")

        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # -------- VALIDATION --------
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for images, targets in val_loader:
                images = [img.to(device) for img in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                model.train()
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                val_loss += losses.item()
                model.eval()

        avg_val_loss = val_loss / len(val_loader)

        lr_scheduler.step()

        val_losses.append(avg_val_loss)

        # -------- PLOTTING --------
        plt.figure(figsize=(8, 6))
        plt.plot(range(start_epoch, epoch + 1), train_losses, label='Train Loss')
        plt.plot(range(start_epoch, epoch + 1), val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training & Validation Loss - MaskRCNN v2')
        plt.legend()
        plt.grid(True)
        plt.savefig("loss_curve_v2_run5.png")
        plt.close()

        print(f"\nEvaluating mAP on validation set for epoch {epoch}...")
        evaluate_mAP(model, val_dataset, device, score_threshold=0.1)

        print(f"Epoch {epoch} - Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")

        # -------- SAVE CHECKPOINT --------
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_checkpoint(
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=lr_scheduler,
                best_val_loss=best_val_loss,
                filename="best_maskrcnn_v2_mushroom_FULL.pth"
            )

if __name__ == "__main__":
    main()
