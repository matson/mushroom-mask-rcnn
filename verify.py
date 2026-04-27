import torch
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from train_model import MushroomCOCODataset

# -------- CONFIG --------
SCORE_THRESHOLD = 0.5
MASK_THRESHOLD  = 0.5
NUM_IMAGES      = 5
CHECKPOINT_PATH = "best_maskrcnn_mushroom_FULL.pth"

CATEGORY_NAMES = {
    1: "Mushrooms",
    2: "BB",
    3: "WB"
}

# One color per class (R, G, B) in 0-1 range
CATEGORY_COLORS = {
    1: (1.0, 0.5, 0.0),   # orange — Mushrooms
    2: (0.0, 0.8, 0.2),   # green  — BB
    3: (0.2, 0.6, 1.0),   # blue   — WB
}

# -------- DEVICE --------
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# -------- DATASET --------
test_dataset = MushroomCOCODataset(
    images_dir=r"C:\data\M18KV2\test\rgb",
    annotations_file=r"C:\data\M18KV2\test\annotations_coco.json",
    augmentations=None,
    resize=(640, 640)
)

# -------- LOAD MODEL --------
num_classes = 4

weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
model = maskrcnn_resnet50_fpn(
    weights=weights,
    rpn_post_nms_top_n_train=500,
    box_detections_per_img=220
)

in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, 256, num_classes)

model.to(device)

checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
print(f"Loaded checkpoint: {CHECKPOINT_PATH}")

# -------- INFERENCE & PLOTTING --------
indices = random.sample(range(len(test_dataset)), NUM_IMAGES)

fig, axes = plt.subplots(1, NUM_IMAGES, figsize=(6 * NUM_IMAGES, 8))

for ax, idx in zip(axes, indices):
    img, target = test_dataset[idx]

    with torch.no_grad():
        pred = model([img.to(device)])[0]

    img_np = img.permute(1, 2, 0).numpy()
    ax.imshow(img_np)

    boxes  = pred['boxes'].cpu().numpy()
    scores = pred['scores'].cpu().numpy()
    labels = pred['labels'].cpu().numpy()
    masks  = pred['masks'].cpu().numpy()  # N x 1 x H x W

    for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
        if score < SCORE_THRESHOLD:
            continue
        if label == 0:
            continue

        color = CATEGORY_COLORS.get(int(label), (1.0, 1.0, 0.0))
        name  = CATEGORY_NAMES.get(int(label), "unknown")

        # Draw bounding box
        x1, y1, x2, y2 = box
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2, edgecolor=color, facecolor='none'
        )
        ax.add_patch(rect)

        # Draw label + score
        ax.text(
            x1, y1 - 4,
            f"{name} {score:.2f}",
            color=color,
            fontsize=7,
            fontweight='bold',
            bbox=dict(facecolor='black', alpha=0.4, pad=1, edgecolor='none')
        )

        # Overlay mask
        binary_mask = masks[i, 0] > MASK_THRESHOLD
        overlay = np.zeros((*binary_mask.shape, 4))  # RGBA
        overlay[binary_mask] = [*color, 0.45]
        ax.imshow(overlay)

    ax.axis('off')
    ax.set_title(f"idx {idx} — {int((scores > SCORE_THRESHOLD).sum())} dets", fontsize=9)

plt.tight_layout()
plt.savefig("predictions_test.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved predictions_test.png")
