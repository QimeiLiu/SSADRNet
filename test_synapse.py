"""Synapse test: offline evaluation with per-class Dice and HD95.

Usage:
  python test_synapse.py --checkpoint '<ckpt_or_glob>' [--img_size 256] [--csv out.csv]
"""

import argparse
import csv
import glob
import os

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from medpy import metric
from scipy.ndimage import label as connected_components
from scipy.ndimage import zoom

from lib.abl import build_model


# Full model (A+B+C), 9 classes, image size 256.
FULL_ARCH = 'a_b_c_unified'

CLASS_NAMES = ['spleen', 'right_kidney', 'left_kidney', 'gallbladder',
               'pancreas', 'liver', 'stomach', 'aorta']


def normalize_synapse_labels(label):
    label = label.copy()
    for c in (5, 9, 10, 12, 13):
        label[label == c] = 0
    label[label == 11] = 5
    return label


def crop_to_union(pred, gt, margin=2):
    union = pred | gt
    if not union.any():
        return pred, gt
    coords = np.where(union)
    slices = []
    for axis, c in enumerate(coords):
        lo = max(int(c.min()) - margin, 0)
        hi = min(int(c.max()) + margin + 1, pred.shape[axis])
        slices.append(slice(lo, hi))
    slices = tuple(slices)
    return pred[slices], gt[slices]


def metric_per_class(pred, gt):
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    if pred.sum() > 0 and gt.sum() > 0:
        pred, gt = crop_to_union(pred, gt)
        return metric.binary.dc(pred, gt), metric.binary.hd95(pred, gt)
    if pred.sum() > 0 and gt.sum() == 0:
        return 1.0, 0.0
    return 0.0, 0.0


def keep_largest_components(pred, num_classes=9):
    cleaned = np.zeros_like(pred)
    structure = np.ones((3, 3, 3), dtype=np.uint8)
    for class_id in range(1, num_classes):
        components, count = connected_components(pred == class_id, structure=structure)
        if count == 0:
            continue
        sizes = np.bincount(components.ravel())
        sizes[0] = 0
        cleaned[components == sizes.argmax()] = class_id
    return cleaned


def remove_small_components(pred, min_size, num_classes=9):
    cleaned = np.zeros_like(pred)
    structure = np.ones((3, 3, 3), dtype=np.uint8)
    for class_id in range(1, num_classes):
        components, count = connected_components(pred == class_id, structure=structure)
        if count == 0:
            continue
        sizes = np.bincount(components.ravel())
        keep = sizes >= min_size
        keep[0] = False
        cleaned[keep[components]] = class_id
    return cleaned


@torch.no_grad()
def predict_volume(model, image, img_size, eval_bs=32):
    depth, h, w = image.shape
    if (h, w) != (img_size, img_size):
        volume = zoom(image, (1, img_size / h, img_size / w), order=3)
    else:
        volume = image

    pred = np.zeros((depth, h, w), dtype=np.int16)
    for start in range(0, depth, eval_bs):
        chunk = volume[start:start + eval_bs]
        inp = torch.from_numpy(chunk).unsqueeze(1).float().cuda()
        logits = sum(model(inp))
        logits = F.interpolate(logits, size=(h, w), mode='bilinear', align_corners=False)
        out = torch.argmax(torch.softmax(logits, dim=1), dim=1)
        pred[start:start + eval_bs] = out.cpu().numpy().astype(np.int16)
    return pred


def evaluate_checkpoint(
        model, ckpt, img_size, eval_bs=32, largest_component=False,
        min_component_size=0, volume_path='data/synapse/test_vol_h5_new'):
    state = torch.load(ckpt, map_location='cpu')
    if state and all(k.startswith('module.') for k in state.keys()):
        state = {k[len('module.'):]: v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    model.eval()

    vols = sorted(glob.glob(os.path.join(volume_path, '*.npy.h5')))
    cls_dice = {c: [] for c in range(1, 9)}
    cls_hd95 = {c: [] for c in range(1, 9)}
    case_dice = []
    case_hd95 = []

    for idx, path in enumerate(vols, 1):
        with h5py.File(path, 'r') as f:
            image = f['image'][:]
            label = normalize_synapse_labels(f['label'][:])
        pred = predict_volume(model, image, img_size, eval_bs=eval_bs)
        if largest_component:
            pred = keep_largest_components(pred)
        elif min_component_size > 0:
            pred = remove_small_components(pred, min_component_size)
        dices, hd95s = [], []
        for c in range(1, 9):
            d, h = metric_per_class(pred == c, label == c)
            cls_dice[c].append(d)
            cls_hd95[c].append(h)
            dices.append(d)
            hd95s.append(h)
        case_dice.append(float(np.mean(dices)))
        case_hd95.append(float(np.mean(hd95s)))
        print('  case %02d/%02d %s Dice %.2f HD95 %.2f' % (
            idx, len(vols), os.path.basename(path), np.mean(dices) * 100.0, np.mean(hd95s)),
            flush=True)

    row = {
        'ckpt': ckpt,
        'mean_dice': float(np.mean(case_dice)) * 100.0,
        'mean_hd95': float(np.mean(case_hd95)),
    }
    for c, name in enumerate(CLASS_NAMES, 1):
        row[f'{name}_dice'] = float(np.mean(cls_dice[c])) * 100.0
        row[f'{name}_hd95'] = float(np.mean(cls_hd95[c]))
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True,
                        help='path (or glob) to one or more .pth checkpoints')
    parser.add_argument('--volume_path', default='data/synapse/test_vol_h5_new',
                        help='dir with test .npy.h5 volumes')
    parser.add_argument('--img_size', type=int, default=256,
                        help='input image size (default: 256)')
    parser.add_argument('--csv', default='', help='optional CSV output path')
    parser.add_argument('--eval_bs', type=int, default=32,
                        help='number of 2D slices per inference batch')
    parser.add_argument('--largest_component', action='store_true')
    parser.add_argument('--min_component_size', type=int, default=0)
    args = parser.parse_args()

    ckpts = sorted(glob.glob(args.checkpoint), key=os.path.getmtime)
    if not ckpts:
        raise SystemExit('NO CKPT for glob: %s' % args.checkpoint)

    model = build_model(FULL_ARCH, 9, args.img_size, 'small').cuda()
    rows = []
    for ckpt in ckpts:
        row = evaluate_checkpoint(
            model, ckpt, args.img_size, eval_bs=args.eval_bs,
            largest_component=args.largest_component,
            min_component_size=args.min_component_size,
            volume_path=args.volume_path)
        rows.append(row)
        print('%s | Dice %.2f | HD95 %.2f' % (ckpt, row['mean_dice'], row['mean_hd95']))
        print('  per-class:', ' '.join(
            '%s %.3f/%.2f' % (name, row[f'{name}_dice'] / 100.0, row[f'{name}_hd95'])
            for name in CLASS_NAMES))

    if args.csv:
        with open(args.csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)


if __name__ == '__main__':
    main()
