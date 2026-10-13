"""Base MaxViT-CASCADE model used by the release."""

import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.decoders import CASCADE_Add
from lib.maxxvit_4out import (
    maxvit_rmlp_small_rw_224 as maxvit_rmlp_small_rw_224_4out,
    maxxvit_rmlp_small_rw_256 as maxxvit_rmlp_small_rw_256_4out,
)


def load_pretrained_weights(img_size, model_scale):
    """Create MaxViT-small and optionally load an external pretrained checkpoint."""
    if model_scale != "small" or img_size not in (224, 256):
        raise ValueError("the release supports MaxViT-small at 224 or 256")
    if img_size == 224:
        backbone = maxvit_rmlp_small_rw_224_4out()
        filename = "maxvit_rmlp_small_rw_224_sw-6ef0ae4f.pth"
    else:
        backbone = maxxvit_rmlp_small_rw_256_4out()
        filename = "maxxvit_rmlp_small_rw_256_sw-37e217ff.pth"
    default_path = os.path.join("./pretrained_pth/maxvit", filename)
    path = os.environ.get("MERIT_PRETRAINED", default_path)
    if os.path.isfile(path):
        print("Loading optional pretrained backbone:", path)
        state_dict = torch.load(path, map_location="cpu")
        backbone.load_state_dict(state_dict, strict=False)
    else:
        print("Pretrained backbone not bundled; using initialized backbone.")
    return backbone


class MaxViT_CASCADE(nn.Module):
    def __init__(
            self, n_class=9, img_size=256, model_scale="small",
            decoder_aggregation="additive"):
        super().__init__()
        if decoder_aggregation != "additive":
            raise ValueError("the release package supports additive CASCADE only")
        self.n_class = n_class
        self.conv = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=1),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True),
        )
        self.backbone = load_pretrained_weights(img_size, model_scale)
        self.channels = [768, 384, 192, 96]
        self.decoder = CASCADE_Add(channels=self.channels)
        self.out_head1 = nn.Conv2d(self.channels[0], n_class, 1)
        self.out_head2 = nn.Conv2d(self.channels[1], n_class, 1)
        self.out_head3 = nn.Conv2d(self.channels[2], n_class, 1)
        self.out_head4 = nn.Conv2d(self.channels[3], n_class, 1)

    def forward(self, x):
        if x.size(1) == 1:
            x = self.conv(x)
        features = self.backbone(x)
        d4, d3, d2, d1 = self.decoder(
            features[3], [features[2], features[1], features[0]])
        p1 = F.interpolate(self.out_head1(d4), scale_factor=32, mode="bilinear")
        p2 = F.interpolate(self.out_head2(d3), scale_factor=16, mode="bilinear")
        p3 = F.interpolate(self.out_head3(d2), scale_factor=8, mode="bilinear")
        p4 = F.interpolate(self.out_head4(d1), scale_factor=4, mode="bilinear")
        return p1, p2, p3, p4
