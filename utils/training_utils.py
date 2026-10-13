"""Training and validation utilities used by the Synapse release."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from medpy import metric
from scipy.ndimage import zoom


def powerset(sequence):
    if len(sequence) <= 1:
        yield sequence
        yield []
        return
    for item in powerset(sequence[1:]):
        yield [sequence[0]] + item
        yield item


class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.n_classes = n_classes

    def _one_hot(self, target):
        return torch.cat(
            [(target == index).unsqueeze(1) for index in range(self.n_classes)],
            dim=1).float()

    @staticmethod
    def _dice_loss(score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        denominator = torch.sum(target * target) + torch.sum(score * score)
        return 1 - (2 * intersect + smooth) / (denominator + smooth)

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot(target)
        if weight is None:
            weight = [1] * self.n_classes
        if inputs.size() != target.size():
            raise ValueError(
                f"prediction {inputs.size()} and target {target.size()} differ")
        losses = [
            self._dice_loss(inputs[:, index], target[:, index]) * weight[index]
            for index in range(self.n_classes)]
        return sum(losses) / self.n_classes


class FocalLoss(nn.Module):
    def __init__(
            self, gamma=2.0, alpha=None, reduction="mean",
            adaptive=False, num_classes=9):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.adaptive = adaptive
        self.num_classes = num_classes

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        focal_loss = (1 - torch.exp(-ce_loss)) ** self.gamma * ce_loss
        if self.adaptive:
            with torch.no_grad():
                count = torch.bincount(
                    targets.reshape(-1), minlength=self.num_classes).float()
                weights = 1.0 / torch.sqrt(count + 1.0)
                weights = weights / weights.mean()
            focal_loss = weights[targets] * focal_loss
        elif self.alpha is not None:
            alpha = (
                self.alpha[targets]
                if torch.is_tensor(self.alpha) else self.alpha)
            focal_loss = alpha * focal_loss
        if self.reduction == "mean":
            return focal_loss.mean()
        if self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


def calculate_dice(prediction, target):
    prediction = prediction.astype(bool)
    target = target.astype(bool)
    if prediction.any() and target.any():
        return metric.binary.dc(prediction, target)
    if prediction.any() and not target.any():
        return 1
    return 0


def val_single_volume(
        image, label, net, classes, patch_size=(256, 256),
        test_save_path=None, case=None, z_spacing=1):
    del test_save_path, case, z_spacing
    image = image.squeeze(0).cpu().detach().numpy()
    label = label.squeeze(0).cpu().detach().numpy()

    if image.ndim == 3:
        depth, height, width = image.shape
        if (height, width) != tuple(patch_size):
            volume = zoom(
                image,
                (1, patch_size[0] / height, patch_size[1] / width),
                order=3)
        else:
            volume = image
        prediction = np.zeros(
            (depth, patch_size[0], patch_size[1]), dtype=np.int16)
        net.eval()
        with torch.no_grad():
            for start in range(0, depth, 32):
                chunk = torch.from_numpy(
                    volume[start:start + 32]).unsqueeze(1).float().cuda()
                logits = sum(net(chunk))
                prediction[start:start + 32] = torch.argmax(
                    torch.softmax(logits, dim=1), dim=1
                ).cpu().numpy().astype(np.int16)
        if (height, width) != tuple(patch_size):
            prediction = zoom(
                prediction,
                (1, height / patch_size[0], width / patch_size[1]),
                order=0)
    else:
        tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            prediction = torch.argmax(
                torch.softmax(sum(net(tensor)), dim=1), dim=1
            ).squeeze(0).cpu().numpy()

    return [
        calculate_dice(prediction == index, label == index)
        for index in range(1, classes)]
