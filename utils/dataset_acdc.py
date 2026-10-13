"""ACDC NPZ datasets and 2D training augmentation."""

import random
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage import zoom
from torch.utils.data import Dataset
from torchvision.transforms import ElasticTransform, InterpolationMode
from torchvision.transforms import functional as vision_functional


def random_rot_flip(image, label):
    turns = np.random.randint(0, 4)
    axis = np.random.randint(0, 2)
    return (
        np.flip(np.rot90(image, turns), axis=axis).copy(),
        np.flip(np.rot90(label, turns), axis=axis).copy(),
    )


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    return (
        ndimage.rotate(image, angle, order=0, reshape=False),
        ndimage.rotate(label, angle, order=0, reshape=False),
    )


class RandomGenerator:
    def __init__(self, output_size, mode="standard"):
        self.output_size = tuple(output_size)
        self.mode = mode

    def __call__(self, sample):
        image, label = sample["image"], sample["label"]
        probability = random.random()
        if probability > 0.5:
            image, label = random_rot_flip(image, label)
        elif probability > 0.25 and self.mode == "standard":
            image, label = random_rotate(image, label)
        height, width = image.shape
        if (height, width) != self.output_size:
            image = zoom(
                image,
                (self.output_size[0] / height, self.output_size[1] / width),
                order=3)
            label = zoom(
                label,
                (self.output_size[0] / height, self.output_size[1] / width),
                order=0)
        if self.mode in ("mri", "mri_elastic"):
            image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
            label = torch.from_numpy(label.astype(np.float32)).unsqueeze(0)
            height, width = self.output_size
            affine_args = dict(
                angle=random.uniform(-15.0, 15.0),
                translate=[round(random.uniform(-0.08, 0.08) * width),
                           round(random.uniform(-0.08, 0.08) * height)],
                scale=random.uniform(0.8, 1.2), shear=[0.0, 0.0])
            image = vision_functional.affine(
                image, **affine_args, interpolation=InterpolationMode.BILINEAR,
                fill=float(image.min()))
            image = image.clamp(-1.0, 1.0)
            label = vision_functional.affine(
                label, **affine_args, interpolation=InterpolationMode.NEAREST,
                fill=0.0)
            if self.mode == "mri_elastic" and random.random() < 0.25:
                displacement = ElasticTransform.get_params(
                    alpha=[40.0, 40.0], sigma=[5.0, 5.0], size=[height, width])
                image = vision_functional.elastic_transform(
                    image, displacement, interpolation=InterpolationMode.BILINEAR,
                    fill=float(image.min()))
                label = vision_functional.elastic_transform(
                    label, displacement, interpolation=InterpolationMode.NEAREST,
                    fill=0.0)
            if random.random() < 0.5:
                foreground = image > -0.999
                intensity = ((image + 1.0) * 0.5).clamp(0.0, 1.0)
                intensity = intensity.pow(random.uniform(0.7, 1.5))
                intensity = intensity * random.uniform(0.85, 1.15) + random.uniform(-0.05, 0.05)
                image = torch.where(foreground, 2.0 * intensity.clamp(0.0, 1.0) - 1.0, image)
            if random.random() < 0.2:
                image = (image + torch.randn_like(image) * 0.01).clamp(-1.0, 1.0)
            return {"image": image.clamp(-1.0, 1.0), "label": label.squeeze(0).long()}
        return {
            "image": torch.from_numpy(image.astype(np.float32)).unsqueeze(0),
            "label": torch.from_numpy(label.astype(np.int64)),
        }


class ACDCDataset(Dataset):
    """Read train/valid 2D NPZ slices and test 3D NPZ volumes."""

    def __init__(self, base_dir, list_dir, split, transform=None):
        self.base_dir = Path(base_dir)
        self.split = split
        self.transform = transform
        list_path = Path(list_dir) / f"{split}.txt"
        self.samples = [line.strip() for line in list_path.read_text().splitlines()]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        name = self.samples[index]
        path = self.base_dir / name
        if self.split in ("train", "valid"):
            path = self.base_dir / self.split / name
        data = np.load(path)
        sample = {"image": data["img"], "label": data["label"]}
        if self.transform is not None and self.split == "train":
            sample = self.transform(sample)
        sample["case_name"] = name
        return sample
