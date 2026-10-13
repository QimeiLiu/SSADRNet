#!/usr/bin/env python3
"""ACDC test: evaluate a checkpoint with per-class Dice and HD95.

Usage:
  python test_acdc.py --checkpoint <checkpoint.pth> [--csv out.csv]
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from medpy import metric
from scipy.ndimage import label as connected_components
from scipy.ndimage import zoom
from torch.utils.data import DataLoader

from lib.abl import build_model
from utils.dataset_acdc import ACDCDataset


# Full model (A+B+C), 4 classes (background, RV, Myo, LV).
FULL_ARCH = 'a_b_c_unified'

CLASS_NAMES = ("RV", "Myo", "LV")


def normalize_state(state):
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    if state and all(key.startswith("module.") for key in state):
        state = {key[len("module."):]: value for key, value in state.items()}
    return state


def keep_largest_components(prediction, num_classes=4):
    """Keep one 3D anatomical component per foreground class."""
    cleaned = np.zeros_like(prediction)
    structure = np.ones((3, 3, 3), dtype=np.uint8)
    for class_id in range(1, num_classes):
        components, count = connected_components(
            prediction == class_id, structure=structure)
        if count == 0:
            continue
        sizes = np.bincount(components.ravel())
        sizes[0] = 0
        cleaned[components == sizes.argmax()] = class_id
    return cleaned


@torch.inference_mode()
def predict_volume(model, image, image_size, batch_size):
    depth, height, width = image.shape
    volume = image
    if (height, width) != (image_size, image_size):
        volume = zoom(
            image, (1, image_size / height, image_size / width), order=3)
    prediction = np.zeros((depth, height, width), dtype=np.int16)
    for start in range(0, depth, batch_size):
        tensor = torch.from_numpy(
            volume[start:start + batch_size]).unsqueeze(1).float().cuda()
        logits = sum(model(tensor))
        logits = F.interpolate(
            logits, size=(height, width), mode="bilinear", align_corners=False)
        prediction[start:start + batch_size] = torch.argmax(
            logits, dim=1).cpu().numpy().astype(np.int16)
    return prediction


def class_metrics(prediction, label, class_id):
    pred_mask, label_mask = prediction == class_id, label == class_id
    if pred_mask.any() and label_mask.any():
        return metric.binary.dc(pred_mask, label_mask), metric.binary.hd95(pred_mask, label_mask)
    if not pred_mask.any() and not label_mask.any():
        return 1.0, 0.0
    return 0.0, float("nan")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--volume-dir", default="./data/ACDC/test")
    parser.add_argument("--list-dir", default="./data/ACDC/lists_ACDC")
    parser.add_argument("--img-size", type=int, choices=(224, 256), default=224)
    parser.add_argument("--eval-bs", type=int, default=8)
    parser.add_argument("--csv", default="")
    parser.add_argument("--largest-component", action="store_true")
    args = parser.parse_args()

    model = build_model(FULL_ARCH, 4, args.img_size, "small")
    model.load_state_dict(
        normalize_state(torch.load(args.checkpoint, map_location="cpu")),
        strict=True)
    model = model.cuda().eval()
    loader = DataLoader(
        ACDCDataset(args.volume_dir, args.list_dir, "test"),
        batch_size=1, shuffle=False)

    rows = []
    for sample in loader:
        image = sample["image"].squeeze(0).numpy()
        label = sample["label"].squeeze(0).numpy()
        prediction = predict_volume(model, image, args.img_size, args.eval_bs)
        if args.largest_component:
            prediction = keep_largest_components(prediction)
        values = [class_metrics(prediction, label, index) for index in range(1, 4)]
        row = {"case": sample["case_name"][0]}
        for name, (dice, hd95) in zip(CLASS_NAMES, values):
            row[f"{name}_dice"] = dice * 100.0
            row[f"{name}_hd95"] = hd95
        row["mean_dice"] = float(np.mean([value[0] for value in values])) * 100.0
        row["mean_hd95"] = float(np.nanmean([value[1] for value in values]))
        rows.append(row)
        print(f"{row['case']} Dice={row['mean_dice']:.2f} HD95={row['mean_hd95']:.2f}")

    summary = {
        key: float(np.nanmean([row[key] for row in rows]))
        for key in rows[0] if key != "case"}
    print(f"Mean Dice={summary['mean_dice']:.2f} HD95={summary['mean_hd95']:.2f}")
    if args.csv:
        path = Path(args.csv)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
