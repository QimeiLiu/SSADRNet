#!/usr/bin/env python3
"""Train the full SSADRNet model (A+B+C) on the ACDC dataset."""

import argparse
import json
import random
import re
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from torch.utils.data import DataLoader
from tqdm import tqdm

from lib.abl import adamw_param_groups, build_model, mutation_loss
from utils.dataset_acdc import ACDCDataset, RandomGenerator
from utils.training_utils import DiceLoss, powerset


# Full model (A+B+C).
FULL_ARCH = "a_b_c_unified"


def normalize_state(state):
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    if state and all(key.startswith("module.") for key in state):
        state = {key[len("module."):]: value for key, value in state.items()}
    return state


def foreground_deep_loss(predictions, label, ce_loss, dice_loss, auxiliary_weight=0.5):
    """Prioritize the fused prediction and supervise foreground in every head."""
    def segmentation_loss(logits):
        return 0.3 * ce_loss(logits, label) + 0.7 * dice_loss(
            logits, label, weight=[0.0, 4 / 3, 4 / 3, 4 / 3], softmax=True)

    auxiliary = sum(
        weight * segmentation_loss(prediction)
        for weight, prediction in zip((0.1, 0.2, 0.3, 0.4), predictions))
    return segmentation_loss(sum(predictions)) + auxiliary_weight * auxiliary


def predict_slice(model, image, image_size):
    height, width = image.shape
    from scipy.ndimage import zoom
    resized = image
    if (height, width) != (image_size, image_size):
        resized = zoom(image, (image_size / height, image_size / width), order=3)
    tensor = torch.from_numpy(resized).unsqueeze(0).unsqueeze(0).float().cuda()
    with torch.inference_mode():
        prediction = torch.argmax(sum(model(tensor)), dim=1).squeeze(0).cpu().numpy()
    if (height, width) != (image_size, image_size):
        prediction = zoom(
            prediction, (height / image_size, width / image_size), order=0)
    return prediction


def validation_dice(model, loader, image_size, unit="slice"):
    scores = []
    volumes = {}
    model.eval()
    for sample in loader:
        image = sample["image"].squeeze(0).numpy()
        label = sample["label"].squeeze(0).numpy()
        prediction = predict_slice(model, image, image_size)
        class_scores = []
        if unit == "volume":
            name = sample["case_name"][0]
            match = re.fullmatch(r"(case_\d+)_slice(ED|ES)_(\d+)\.npz", name)
            if match is None:
                raise ValueError(f"Cannot group validation slice: {name}")
            key = (match[1], match[2])
            totals = volumes.setdefault(key, np.zeros((3, 2), dtype=np.float64))
        for class_id in range(1, 4):
            pred_mask, label_mask = prediction == class_id, label == class_id
            denominator = pred_mask.sum() + label_mask.sum()
            if unit == "volume":
                totals[class_id - 1] += ((pred_mask & label_mask).sum(), denominator)
            class_scores.append(
                1.0 if denominator == 0 else 2.0 * (pred_mask & label_mask).sum() / denominator)
        scores.append(np.mean(class_scores))
    if unit == "volume":
        scores = [np.mean(np.divide(
            2.0 * totals[:, 0], totals[:, 1],
            out=np.ones(3), where=totals[:, 1] != 0)) for totals in volumes.values()]
    model.train()
    return float(np.mean(scores))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-dir", default="./data/ACDC")
    parser.add_argument("--list-dir", default="./data/ACDC/lists_ACDC")
    parser.add_argument("--output-dir", default="./model_pth/ACDC")
    parser.add_argument("--img-size", type=int, choices=(224, 256), default=224)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--max-epochs", type=int, default=400)
    parser.add_argument("--lr", type=float, default=0.00045)
    parser.add_argument("--weight-decay", type=float, default=0.001)
    parser.add_argument("--no-decay-norm-bias", action="store_true")
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--auxiliary-weight", type=float, default=0.5)
    parser.add_argument("--foreground-ce-weight", type=float, default=1.0)
    parser.add_argument("--eval-interval", type=int, default=5)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2222)
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--lr-schedule", choices=("constant", "cosine"), default="cosine")
    parser.add_argument("--warmup-epochs", type=int, default=20)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--validation-unit", choices=("slice", "volume"), default="volume")
    parser.add_argument("--loss-mode", choices=("mutation", "foreground_deep"), default="foreground_deep")
    parser.add_argument("--augmentation", choices=("standard", "mri", "mri_elastic"), default="mri")
    parser.add_argument("--ema-decay", type=float, default=0.9995)
    parser.add_argument("--ema-average-buffers", action="store_true")
    args = parser.parse_args()
    if not math.isfinite(args.weight_decay) or args.weight_decay < 0:
        parser.error("weight-decay must be finite and nonnegative")
    if not math.isfinite(args.auxiliary_weight) or args.auxiliary_weight < 0:
        parser.error("auxiliary-weight must be finite and nonnegative")
    if not math.isfinite(args.foreground_ce_weight) or args.foreground_ce_weight <= 0:
        parser.error("foreground-ce-weight must be finite and positive")
    if not 0 <= args.label_smoothing < 1:
        parser.error("label-smoothing must be between 0 and 1 (exclusive)")
    if not 0 <= args.ema_decay < 1:
        parser.error("ema-decay must be between 0 (disabled) and 1 (exclusive)")
    if args.lr_schedule == "cosine" and not 0 <= args.warmup_epochs < args.max_epochs:
        parser.error("warmup-epochs must be between 0 and max-epochs - 1")
    if args.lr_schedule == "cosine" and not 0 <= args.min_lr <= args.lr:
        parser.error("min-lr must be between 0 and lr")

    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    model = build_model(FULL_ARCH, 4, args.img_size, "small").cuda()
    if args.checkpoint:
        model.load_state_dict(
            normalize_state(torch.load(args.checkpoint, map_location="cpu")),
            strict=True)

    train_data = ACDCDataset(
        args.root_dir, args.list_dir, "train",
        RandomGenerator((args.img_size, args.img_size), mode=args.augmentation))
    valid_data = ACDCDataset(args.root_dir, args.list_dir, "valid")
    train_loader = DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True)
    valid_loader = DataLoader(valid_data, batch_size=1, shuffle=False)

    output = Path(args.output_dir) / (
        f"full_ACDC{args.img_size}_seed{args.seed}_{time.strftime('%H%M%S')}")
    output.mkdir(parents=True, exist_ok=True)
    (output / "args.json").write_text(json.dumps(vars(args), indent=2) + "\n")

    optimizer = AdamW(adamw_param_groups(
        model, args.weight_decay, no_decay_norm_bias=args.no_decay_norm_bias), lr=args.lr)
    ema_model = (
        AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(args.ema_decay),
                      use_buffers=args.ema_average_buffers)
        if args.ema_decay else None)
    ce_weights = (
        torch.tensor([1.0] + [args.foreground_ce_weight] * 3, device="cuda")
        if args.foreground_ce_weight != 1.0 else None)
    ce_loss = CrossEntropyLoss(weight=ce_weights, label_smoothing=args.label_smoothing)
    dice_loss = DiceLoss(4)
    subsets = list(powerset([0, 1, 2, 3]))
    best = -1.0
    for epoch in tqdm(range(args.max_epochs), ncols=80):
        if args.lr_schedule == "cosine":
            if epoch < args.warmup_epochs:
                learning_rate = args.lr * (epoch + 1) / args.warmup_epochs
            else:
                progress = (epoch - args.warmup_epochs) / max(
                    1, args.max_epochs - args.warmup_epochs - 1)
                learning_rate = args.min_lr + 0.5 * (args.lr - args.min_lr) * (
                    1.0 + math.cos(math.pi * progress))
            for group in optimizer.param_groups:
                group["lr"] = learning_rate
        model.train()
        running = 0.0
        for sample in train_loader:
            image = sample["image"].cuda(non_blocking=True).float()
            label = sample["label"].cuda(non_blocking=True).long()
            predictions = model(image)
            if args.loss_mode == "foreground_deep":
                loss = foreground_deep_loss(
                    predictions, label, ce_loss, dice_loss, args.auxiliary_weight)
            else:
                loss = mutation_loss(
                    predictions, label, ce_loss, dice_loss, subsets,
                    lc1=0.3, lc2=0.7)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            if ema_model is not None:
                ema_model.update_parameters(model)
            running += loss.item()
        torch.save(model.state_dict(), output / "last.pth")
        if (epoch + 1) % args.eval_interval == 0 or epoch + 1 == args.max_epochs:
            score = validation_dice(model, valid_loader, args.img_size, args.validation_unit)
            raw_score = score
            selected_model, selected_source = model, "raw"
            ema_log = ""
            if ema_model is not None:
                ema_score = validation_dice(
                    ema_model, valid_loader, args.img_size, args.validation_unit)
                ema_log = f" ema_dice={ema_score:.5f} raw_dice={raw_score:.5f}"
                if ema_score > score:
                    score = ema_score
                    selected_model, selected_source = ema_model.module, "ema"
            print(f"epoch={epoch + 1} loss={running / len(train_loader):.5f} "
                  f"val_dice={score:.5f} val_unit={args.validation_unit} "
                  f"lr={optimizer.param_groups[0]['lr']:.8f}{ema_log}")
            if score >= best:
                best = score
                torch.save(selected_model.state_dict(), output / "best.pth")
                (output / "best_selection.json").write_text(json.dumps({
                    "epoch": epoch + 1, "validation_dice": score,
                    "validation_unit": args.validation_unit,
                    "source": selected_source,
                }, indent=2) + "\n")
    print(f"completed best_val_dice={best:.5f} output={output}")


if __name__ == "__main__":
    main()
