import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import os
import copy
from collections import Counter
from PIL import Image
import torch
from torch.utils.data import Dataset
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np


def evaluate_mAP(model, val_dataset, device, score_threshold=0.1):
    """
    Compute COCO-style mAP for bbox and segmentation (mask) predictions.
    Evaluates per-class: Mushrooms (1), BB (2), WB (3).

    Category mapping:
        model label 1 → category_id 1 (Mushrooms)
        model label 2 → category_id 2 (BB)
        model label 3 → category_id 3 (WB)

    GT category_ids are remapped from 0,1,2 → 1,2,3 to avoid pycocotools
    issues with category_id=0.
    """
    if isinstance(val_dataset, torch.utils.data.Subset):
        base_dataset = val_dataset.dataset
    else:
        base_dataset = val_dataset

    model.eval()
    bbox_results = []
    segm_results = []

    print("Running inference on validation set...")
    with torch.no_grad():
        for idx in range(len(val_dataset)):
            img, target = val_dataset[idx]
            img = img.to(device)

            pred = model([img])[0]

            boxes   = pred['boxes'].cpu().numpy()
            scores  = pred['scores'].cpu().numpy()
            labels  = pred['labels'].cpu().numpy()
            masks   = pred['masks'].cpu().numpy()  # N x 1 x H x W, float [0,1]

            img_id = int(target['image_id'].item())
            img_info = base_dataset.coco.loadImgs(img_id)[0]
            w_orig, h_orig = img_info['width'], img_info['height']
            w_new, h_new = base_dataset.resize
            scale_x = w_orig / w_new
            scale_y = h_orig / h_new

            for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
                if score < score_threshold:
                    continue
                if label == 0:
                    continue

                x1, y1, x2, y2 = box
                x1 *= scale_x; x2 *= scale_x
                y1 *= scale_y; y2 *= scale_y
                width  = max(0, x2 - x1)
                height = max(0, y2 - y1)

                # label maps directly to category_id (1,2,3)
                category_id = int(label)

                bbox_results.append({
                    "image_id": img_id,
                    "category_id": category_id,
                    "bbox": [float(x1), float(y1), float(width), float(height)],
                    "score": float(score)
                })

                # Convert mask to RLE for segmentation eval
                from pycocotools import mask as mask_utils
                binary_mask = (masks[i, 0] > 0.5).astype(np.uint8)
                # Resize mask back to original image size
                mask_img = Image.fromarray(binary_mask)
                mask_img = mask_img.resize((w_orig, h_orig), resample=Image.NEAREST)
                binary_mask_orig = np.asfortranarray(np.array(mask_img))
                rle = mask_utils.encode(binary_mask_orig)
                rle['counts'] = rle['counts'].decode('utf-8')

                segm_results.append({
                    "image_id": img_id,
                    "category_id": category_id,
                    "segmentation": rle,
                    "score": float(score)
                })

    if len(bbox_results) == 0:
        print("No predictions above threshold!")
        return

    # Remap GT category_ids: 0→1, 1→2, 2→3 (avoid category_id=0 in pycocotools)
    cocoGt = copy.deepcopy(base_dataset.coco)
    for ann in cocoGt.dataset['annotations']:
        ann['category_id'] = ann['category_id'] + 1
    cocoGt.dataset['categories'] = [
        {"id": 1, "name": "Mushrooms"},
        {"id": 2, "name": "BB"},
        {"id": 3, "name": "WB"}
    ]
    cocoGt.createIndex()

    counts = Counter(ann['image_id'] for ann in cocoGt.dataset['annotations'])
    max_dets = max(counts.values())

    # --- Bounding box mAP ---
    print("\n--- Bounding Box mAP ---")
    cocoDt_bbox = cocoGt.loadRes(bbox_results)
    cocoEval_bbox = COCOeval(cocoGt, cocoDt_bbox, iouType='bbox')
    cocoEval_bbox.params.maxDets = [1, 10, max_dets]
    cocoEval_bbox.evaluate()
    cocoEval_bbox.accumulate()
    cocoEval_bbox.summarize()
    print(f"BBox AP50: {cocoEval_bbox.stats[1]:.4f}")

    # --- Segmentation mAP ---
    print("\n--- Segmentation mAP ---")
    cocoDt_segm = cocoGt.loadRes(segm_results)
    cocoEval_segm = COCOeval(cocoGt, cocoDt_segm, iouType='segm')
    cocoEval_segm.params.maxDets = [1, 10, max_dets]
    cocoEval_segm.evaluate()
    cocoEval_segm.accumulate()
    cocoEval_segm.summarize()
    print(f"Segm AP50: {cocoEval_segm.stats[1]:.4f}")


def save_checkpoint(epoch, model, optimizer, scheduler, best_val_loss, filename="checkpoint.pth"):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_val_loss": best_val_loss
    }
    torch.save(checkpoint, filename)
    print(f"Checkpoint saved: {filename}")


def load_checkpoint(filename, model, optimizer=None, scheduler=None, device="cpu"):
    checkpoint = torch.load(filename, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer and scheduler:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    epoch = checkpoint["epoch"]
    best_val_loss = checkpoint["best_val_loss"]
    print(f"Checkpoint loaded: {filename} (epoch {epoch})")
    return epoch, best_val_loss


def visualize_samples(dataset, num_samples=3):
    samples = [dataset[i] for i in range(num_samples)]

    plt.figure(figsize=(12, 4 * num_samples))

    for i, (img, target) in enumerate(samples):
        img_np = img.permute(1, 2, 0).numpy()

        plt.subplot(num_samples, 1, i + 1)
        plt.imshow(img_np)
        plt.axis('off')

        for box in target['boxes']:
            x1, y1, x2, y2 = box.numpy()
            plt.gca().add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1,
                                          fill=False, color='red', linewidth=2))

        for mask in target['masks']:
            mask_np = mask.numpy()
            plt.imshow(mask_np, alpha=0.4)

    plt.tight_layout()
    plt.show()
