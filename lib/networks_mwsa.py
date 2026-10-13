"""Single-backbone (MaxViT-small) + CASCADE decoder variants for the 224 Synapse study.

Kept minimal on purpose: only the current working base ('base_msdec' = base + multi-scale
conv decoder, the best single-backbone-224 model at 83.64 DICE). New single-change experiments
should be added one at a time and pruned once evaluated.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import deform_conv2d

from lib.networks import MaxViT_CASCADE


class DeformMSBlock(nn.Module):
    """Self-designed Deformable Multi-Scale block (single change on top of base_msdec).
    Content-adaptive sampling at multiple kernel scales {3,5,7}: each scale predicts its own
    offset field and applies a deformable conv, then fused + projected. Targets irregular small
    organs (gallbladder/pancreas). Uses torchvision deform_conv2d (no natten / no CUDA build).

    Init discipline (avoids the dead-branch trap): gate zero-init -> block starts == identity
    (floor = base_msdec 83.64); offset convs zero-init -> starts UNdeformed (regular conv);
    deform-conv weights normal-init so the fused output is non-zero and the gate receives a
    real gradient and can lift off."""

    def __init__(self, ch, kernels=(3, 5, 7)):
        super().__init__()
        self.kernels = kernels
        # DEPTHWISE deform conv (weight [ch,1,k,k]) -> lightweight; spatial adaptivity is the point,
        # not capacity. Offset predicted by a depthwise-separable conv (cheap), last layer zero-init
        # so it starts UNdeformed. deform weight normal-init so fused != 0 and the gate can lift off.
        self.off_dw = nn.ModuleList([nn.Conv2d(ch, ch, k, padding=k // 2, groups=ch, bias=False) for k in kernels])
        self.off_pw = nn.ModuleList([nn.Conv2d(ch, 2 * k * k, 1) for k in kernels])
        self.weight = nn.ParameterList([nn.Parameter(torch.empty(ch, 1, k, k)) for k in kernels])
        for w in self.weight:
            nn.init.kaiming_normal_(w, nonlinearity='relu')
        for pw in self.off_pw:                                  # start undeformed (offset == 0)
            nn.init.zeros_(pw.weight); nn.init.zeros_(pw.bias)
        self.proj = nn.Sequential(nn.Conv2d(ch * len(kernels), ch, 1, bias=False), nn.BatchNorm2d(ch))
        self.act = nn.GELU()
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        outs = []
        for i, k in enumerate(self.kernels):
            off = self.off_pw[i](self.off_dw[i](x))            # [B, 2*k*k, H, W]
            outs.append(deform_conv2d(x, off, self.weight[i], padding=k // 2))  # depthwise (groups=ch)
        fused = self.proj(self.act(torch.cat(outs, 1)))
        return x + self.gate * fused


class Single224_CASCADE_MSDec_DMS(nn.Module):
    """base_msdec + DeformMSBlock on each decoder output (single change: deformable multi-scale).
    zero-init gate -> starts == base_msdec. Single backbone, 224."""

    def __init__(self, n_class=9, img_size=224, model_scale='small'):
        super().__init__()
        from lib.networks import load_pretrained_weights
        from lib.decoders import CASCADE_Add
        self.conv = nn.Sequential(nn.Conv2d(1, 3, 1), nn.BatchNorm2d(3), nn.ReLU(inplace=True))
        self.backbone = load_pretrained_weights(img_size, model_scale)
        self.channels = [768, 384, 192, 96] if model_scale == 'small' else [512, 256, 128, 64]
        self.decoder = CASCADE_Add(channels=self.channels)
        self.ms = nn.ModuleList([MSConvBlock(c) for c in self.channels])
        self.dms = nn.ModuleList([DeformMSBlock(c) for c in self.channels])
        self.out_head1 = nn.Conv2d(self.channels[0], n_class, 1)
        self.out_head2 = nn.Conv2d(self.channels[1], n_class, 1)
        self.out_head3 = nn.Conv2d(self.channels[2], n_class, 1)
        self.out_head4 = nn.Conv2d(self.channels[3], n_class, 1)

    def forward(self, x):
        if x.size(1) == 1:
            x = self.conv(x)
        f = list(self.backbone(x))
        d4, d3, d2, d1 = self.decoder(f[3], [f[2], f[1], f[0]])
        ds = [self.ms[0](d4), self.ms[1](d3), self.ms[2](d2), self.ms[3](d1)]
        d4, d3, d2, d1 = [self.dms[i](ds[i]) for i in range(4)]
        p1 = F.interpolate(self.out_head1(d4), scale_factor=32, mode='bilinear', align_corners=False)
        p2 = F.interpolate(self.out_head2(d3), scale_factor=16, mode='bilinear', align_corners=False)
        p3 = F.interpolate(self.out_head3(d2), scale_factor=8, mode='bilinear', align_corners=False)
        p4 = F.interpolate(self.out_head4(d1), scale_factor=4, mode='bilinear', align_corners=False)
        return p1, p2, p3, p4


class DeformRefine(nn.Module):
    """Absorb AgileFormer's DAT-style deformable attention (cited, blocks_2d.DAttentionBaseline) as
    a zero-init residual refinement on a MaxViT backbone-stage feature [B,C,H,W]. Spatial-adaptive
    sampling = the mechanism proven to lift small/irregular organs (pancreas). gate zero-init ->
    starts == identity (floor=base_msdec); applied at the ENCODER feature (where deformable works),
    not the decoder tail (where our earlier crude deform_conv failed)."""

    def __init__(self, ch, fmap, ksize=5):
        super().__init__()
        from lib.agf.blocks_2d import DAttentionBaseline, LayerNormProxy
        nh = max(1, ch // 32)
        ng = max(1, nh // 2)
        self.norm = LayerNormProxy(ch)
        self.dat = DAttentionBaseline((fmap, fmap), (fmap, fmap), nh, 32, ng,
                                      0.0, 0.0, 1, 2, True, False, False, False, ksize, False)
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        y, _, _ = self.dat(self.norm(x))
        return x + self.gate * y


class Single224_CASCADE_MSDec_DAT(nn.Module):
    """OUR base_msdec + absorbed AgileFormer deformable attention (DeformRefine) on backbone
    stages f1,f2,f3 (28/14/7). zero-init -> starts == base_msdec (83.64). Single MaxViT bb, 224.
    Tests whether the proven deformable mechanism, applied to OUR encoder features, lifts us."""

    def __init__(self, n_class=9, img_size=224, model_scale='small'):
        super().__init__()
        from lib.networks import load_pretrained_weights
        from lib.decoders import CASCADE_Add
        self.conv = nn.Sequential(nn.Conv2d(1, 3, 1), nn.BatchNorm2d(3), nn.ReLU(inplace=True))
        self.backbone = load_pretrained_weights(img_size, model_scale)
        self.channels = [768, 384, 192, 96] if model_scale == 'small' else [512, 256, 128, 64]
        fm = img_size // 32                                    # 7 at 224
        # deformable refine on f2(14,384),f3(7,768); skip f0(56)/f1(28) = grid_sample too slow there
        self.dr1 = None
        self.dr2 = DeformRefine(384, fm * 2, ksize=5)
        self.dr3 = DeformRefine(768, fm * 1, ksize=3)
        self.decoder = CASCADE_Add(channels=self.channels)
        self.ms = nn.ModuleList([MSConvBlock(c) for c in self.channels])
        self.out_head1 = nn.Conv2d(self.channels[0], n_class, 1)
        self.out_head2 = nn.Conv2d(self.channels[1], n_class, 1)
        self.out_head3 = nn.Conv2d(self.channels[2], n_class, 1)
        self.out_head4 = nn.Conv2d(self.channels[3], n_class, 1)

    def forward(self, x):
        if x.size(1) == 1:
            x = self.conv(x)
        f0, f1, f2, f3 = self.backbone(x)
        f2, f3 = self.dr2(f2), self.dr3(f3)
        d4, d3, d2, d1 = self.decoder(f3, [f2, f1, f0])
        d4, d3, d2, d1 = self.ms[0](d4), self.ms[1](d3), self.ms[2](d2), self.ms[3](d1)
        p1 = F.interpolate(self.out_head1(d4), scale_factor=32, mode='bilinear', align_corners=False)
        p2 = F.interpolate(self.out_head2(d3), scale_factor=16, mode='bilinear', align_corners=False)
        p3 = F.interpolate(self.out_head3(d2), scale_factor=8, mode='bilinear', align_corners=False)
        p4 = F.interpolate(self.out_head4(d1), scale_factor=4, mode='bilinear', align_corners=False)
        return p1, p2, p3, p4


class Single224_CASCADE_MSDec_DATin(nn.Module):
    """DAT deformable attention injected INSIDE the MaxViT backbone (between stages), so the
    deformed features PROPAGATE through subsequent backbone stages (true spatial-adaptive feature
    EXTRACTION, vs base_msdec_dat which only refined the backbone OUTPUTS). Backbone trained
    end-to-end (NOT frozen). zero-init gate -> starts == base_msdec. Single MaxViT bb, 224."""

    def __init__(self, n_class=9, img_size=224, model_scale='small'):
        super().__init__()
        from lib.networks import load_pretrained_weights
        from lib.decoders import CASCADE_Add
        self.conv = nn.Sequential(nn.Conv2d(1, 3, 1), nn.BatchNorm2d(3), nn.ReLU(inplace=True))
        self.backbone = load_pretrained_weights(img_size, model_scale)
        self.channels = [768, 384, 192, 96] if model_scale == 'small' else [512, 256, 128, 64]
        fm = img_size // 32                                    # 7
        # inject between stage1->2 (28x28,192) and stage2->3 (14x14,384): influences stages 2,3
        self.drm2 = DeformRefine(192, fm * 4, ksize=7)
        self.drm3 = DeformRefine(384, fm * 2, ksize=5)
        self.decoder = CASCADE_Add(channels=self.channels)
        self.ms = nn.ModuleList([MSConvBlock(c) for c in self.channels])
        self.out_head1 = nn.Conv2d(self.channels[0], n_class, 1)
        self.out_head2 = nn.Conv2d(self.channels[1], n_class, 1)
        self.out_head3 = nn.Conv2d(self.channels[2], n_class, 1)
        self.out_head4 = nn.Conv2d(self.channels[3], n_class, 1)

    def forward(self, x):
        if x.size(1) == 1:
            x = self.conv(x)
        s = self.backbone.stem(x)
        s = self.backbone.stages[0](s); f0 = s
        s = self.backbone.stages[1](s); f1 = s
        s = self.drm2(s)                                       # deform BEFORE stage2
        s = self.backbone.stages[2](s); f2 = s
        s = self.drm3(s)                                       # deform BEFORE stage3
        s = self.backbone.stages[3](s); f3 = self.backbone.norm(s)   # 4-out applies norm to last stage
        d4, d3, d2, d1 = self.decoder(f3, [f2, f1, f0])
        d4, d3, d2, d1 = self.ms[0](d4), self.ms[1](d3), self.ms[2](d2), self.ms[3](d1)
        p1 = F.interpolate(self.out_head1(d4), scale_factor=32, mode='bilinear', align_corners=False)
        p2 = F.interpolate(self.out_head2(d3), scale_factor=16, mode='bilinear', align_corners=False)
        p3 = F.interpolate(self.out_head3(d2), scale_factor=8, mode='bilinear', align_corners=False)
        p4 = F.interpolate(self.out_head4(d1), scale_factor=4, mode='bilinear', align_corners=False)
        return p1, p2, p3, p4


class Single224_CASCADE_SKDec_DATin(Single224_CASCADE_MSDec_DATin):
    """PROPER innovation 1+2: deformable injection (drm2/drm3, innov2) + SELECTIVE multi-scale
    decoder (MSConvBlockSK, fixed innov1) — instead of the old plain MSConvBlock which dragged the
    combination down. innov1(decoder-side) and innov2(encoder-side) attack different stages, so this
    is the additive combination. Same deformable forward as parent; only self.ms swapped to SK."""

    def __init__(self, n_class=9, img_size=224, model_scale='small'):
        super().__init__(n_class=n_class, img_size=img_size, model_scale=model_scale)
        self.ms = nn.ModuleList([MSConvBlockSK(c) for c in self.channels])


class Single224_BigBB_MSDec(nn.Module):
    """STRONGER pretrained backbone (timm, e.g. maxvit_base_tf_224, 118M) + our CASCADE decoder +
    multi-scale conv refinement (MSConvBlock). Single backbone, 224. Adds CAPACITY via a stronger
    pretrained model (user-allowed) while keeping our decoder innovation. Backbone channels must be
    [96,192,384,768] at strides /4../32 (maxvit_base matches base_msdec's decoder exactly)."""

    def __init__(self, n_class=9, img_size=224, backbone='maxvit_base_tf_224'):
        super().__init__()
        import timm
        from lib.decoders import CASCADE_Add
        self.conv = nn.Sequential(nn.Conv2d(1, 3, 1), nn.BatchNorm2d(3), nn.ReLU(inplace=True))
        self.backbone = timm.create_model(backbone, pretrained=True, features_only=True,
                                          out_indices=(1, 2, 3, 4))
        ch = self.backbone.feature_info.channels()                  # [96,192,384,768] ascending
        self.channels = ch[::-1]                                    # [768,384,192,96]
        self.decoder = CASCADE_Add(channels=self.channels)
        self.ms = nn.ModuleList([MSConvBlock(c) for c in self.channels])
        self.out_head1 = nn.Conv2d(self.channels[0], n_class, 1)
        self.out_head2 = nn.Conv2d(self.channels[1], n_class, 1)
        self.out_head3 = nn.Conv2d(self.channels[2], n_class, 1)
        self.out_head4 = nn.Conv2d(self.channels[3], n_class, 1)

    def forward(self, x):
        if x.size(1) == 1:
            x = self.conv(x)
        f = list(self.backbone(x))                                  # [f0../4 .. f3//32]
        d4, d3, d2, d1 = self.decoder(f[3], [f[2], f[1], f[0]])
        d4, d3, d2, d1 = self.ms[0](d4), self.ms[1](d3), self.ms[2](d2), self.ms[3](d1)
        p1 = F.interpolate(self.out_head1(d4), scale_factor=32, mode='bilinear', align_corners=False)
        p2 = F.interpolate(self.out_head2(d3), scale_factor=16, mode='bilinear', align_corners=False)
        p3 = F.interpolate(self.out_head3(d2), scale_factor=8, mode='bilinear', align_corners=False)
        p4 = F.interpolate(self.out_head4(d1), scale_factor=4, mode='bilinear', align_corners=False)
        return p1, p2, p3, p4


class Single224_MaxViT_GCASCADE(nn.Module):
    """OUR single MaxViT-small backbone + G-CASCADE graph-convolution decoder (Rahman & Marculescu,
    WACV 2024 — cited, integrated as a stronger-decoder single change vs our CASCADE/MSConvBlock).
    Graph conv (MRConv + dilated kNN) captures long-range context; reported +2.22 over CASCADE on
    single PVT. Single backbone, 224. p1-p4 -> our MUTATION loss, same as base_msdec."""

    def __init__(self, n_class=9, img_size=224, model_scale='small', skip_aggregation='additive'):
        super().__init__()
        from lib.networks import load_pretrained_weights
        from lib.gcascade_decoder import GCASCADE, GCASCADE_Cat
        self.conv = nn.Sequential(nn.Conv2d(1, 3, 1), nn.BatchNorm2d(3), nn.ReLU(inplace=True))
        self.backbone = load_pretrained_weights(img_size, model_scale)
        self.channels = [768, 384, 192, 96] if model_scale == 'small' else [512, 256, 128, 64]
        if skip_aggregation == 'concatenation':
            self.decoder = GCASCADE_Cat(channels=self.channels, img_size=img_size, k=11, padding=5,
                                        conv='mr', gcb_act='gelu', activation='relu')
            self.channels = [self.channels[0], self.channels[1] * 2, self.channels[2] * 2, self.channels[3] * 2]
        else:
            self.decoder = GCASCADE(channels=self.channels, img_size=img_size, k=11, padding=5,
                                    conv='mr', gcb_act='gelu', activation='relu')
        self.out_head1 = nn.Conv2d(self.channels[0], n_class, 1)
        self.out_head2 = nn.Conv2d(self.channels[1], n_class, 1)
        self.out_head3 = nn.Conv2d(self.channels[2], n_class, 1)
        self.out_head4 = nn.Conv2d(self.channels[3], n_class, 1)

    def forward(self, x):
        if x.size(1) == 1:
            x = self.conv(x)
        f = list(self.backbone(x))
        x1_o, x2_o, x3_o, x4_o = self.decoder(f[3], [f[2], f[1], f[0]])
        p1 = F.interpolate(self.out_head1(x1_o), scale_factor=32, mode='bilinear', align_corners=False)
        p2 = F.interpolate(self.out_head2(x2_o), scale_factor=16, mode='bilinear', align_corners=False)
        p3 = F.interpolate(self.out_head3(x3_o), scale_factor=8, mode='bilinear', align_corners=False)
        p4 = F.interpolate(self.out_head4(x4_o), scale_factor=4, mode='bilinear', align_corners=False)
        return p1, p2, p3, p4


class MSConvBlock(nn.Module):
    """Multi-Scale Conv residual on a decoder feature: parallel depthwise convs at kernels
    {1,3,5,7} (different receptive fields) summed + projected, added as a zero-init ReZero
    residual (starts == base). This is the +0.41 lever (base 83.23 -> base_msdec 83.64)."""

    def __init__(self, ch, kernels=(1, 3, 5, 7)):
        super().__init__()
        self.dw = nn.ModuleList([
            nn.Conv2d(ch, ch, k, padding=k // 2, groups=ch, bias=False) for k in kernels])
        self.proj = nn.Sequential(nn.Conv2d(ch, ch, 1, bias=False), nn.BatchNorm2d(ch))
        self.act = nn.GELU()
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        ms = self.act(sum(d(x) for d in self.dw))
        return x + self.gate * self.proj(ms)


class MSConvBlockSK(nn.Module):
    """Selective multi-scale conv (fixed innovation-1). Parallel depthwise convs {1,3,5,7}, each
    with its OWN BatchNorm, then fused by per-channel softmax attention over the scales (SK-Net,
    Li et al. CVPR2019) instead of a plain SUM. Rationale: at 256 the higher resolution already
    resolves fine detail, so an unconditional sum of all 4 receptive fields is partly redundant and
    over-fits (plain MSConvBlock = -0.32 vs base@256). Softmax lets the network drive redundant
    scales to ~0 weight per channel -> keeps the useful receptive field, drops the harmful capacity.
    zero-init gate -> starts == base (can't regress at init)."""

    def __init__(self, ch, kernels=(1, 3, 5, 7), r=8):
        super().__init__()
        self.K = len(kernels)
        self.dw = nn.ModuleList([
            nn.Sequential(nn.Conv2d(ch, ch, k, padding=k // 2, groups=ch, bias=False),
                          nn.BatchNorm2d(ch)) for k in kernels])
        d = max(ch // r, 16)
        self.fc = nn.Sequential(nn.Conv2d(ch, d, 1, bias=False), nn.BatchNorm2d(d),
                                nn.ReLU(inplace=True))
        self.att = nn.Conv2d(d, ch * self.K, 1)                # per-channel weight for each scale
        self.proj = nn.Sequential(nn.Conv2d(ch, ch, 1, bias=False), nn.BatchNorm2d(ch))
        self.act = nn.GELU()
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C = x.size(0), x.size(1)
        feats = torch.stack([branch(x) for branch in self.dw], dim=1)   # [B,K,C,H,W]
        u = feats.sum(dim=1)                                            # [B,C,H,W]
        s = self.fc(u.mean(dim=(2, 3), keepdim=True))                   # [B,d,1,1]
        w = self.att(s).view(B, self.K, C, 1, 1)                        # [B,K,C,1,1]
        w = torch.softmax(w, dim=1)                                     # convex per-channel over scales
        v = (feats * w).sum(dim=1)                                      # selective fusion
        return x + self.gate * self.proj(self.act(v))


class Single224_CASCADE_MSDec_SK(nn.Module):
    """FIXED innovation-1: single MaxViT backbone + CASCADE decoder + SELECTIVE multi-scale conv
    refinement (MSConvBlockSK) on each of the 4 decoder outputs. No deformable (isolates innov-1).
    Single backbone."""

    def __init__(self, n_class=9, img_size=224, model_scale='small'):
        super().__init__()
        from lib.networks import load_pretrained_weights
        from lib.decoders import CASCADE_Add
        self.conv = nn.Sequential(nn.Conv2d(1, 3, 1), nn.BatchNorm2d(3), nn.ReLU(inplace=True))
        self.backbone = load_pretrained_weights(img_size, model_scale)
        self.channels = [768, 384, 192, 96] if model_scale == 'small' else [512, 256, 128, 64]
        self.decoder = CASCADE_Add(channels=self.channels)
        self.ms = nn.ModuleList([MSConvBlockSK(c) for c in self.channels])
        self.out_head1 = nn.Conv2d(self.channels[0], n_class, 1)
        self.out_head2 = nn.Conv2d(self.channels[1], n_class, 1)
        self.out_head3 = nn.Conv2d(self.channels[2], n_class, 1)
        self.out_head4 = nn.Conv2d(self.channels[3], n_class, 1)

    def forward(self, x):
        if x.size(1) == 1:
            x = self.conv(x)
        f = list(self.backbone(x))
        d4, d3, d2, d1 = self.decoder(f[3], [f[2], f[1], f[0]])
        d4, d3, d2, d1 = self.ms[0](d4), self.ms[1](d3), self.ms[2](d2), self.ms[3](d1)
        p1 = F.interpolate(self.out_head1(d4), scale_factor=32, mode='bilinear', align_corners=False)
        p2 = F.interpolate(self.out_head2(d3), scale_factor=16, mode='bilinear', align_corners=False)
        p3 = F.interpolate(self.out_head3(d2), scale_factor=8, mode='bilinear', align_corners=False)
        p4 = F.interpolate(self.out_head4(d1), scale_factor=4, mode='bilinear', align_corners=False)
        return p1, p2, p3, p4


class Single224_CASCADE_DATin_NoMS(nn.Module):
    """STANDALONE innovation-2: DAT deformable injection INSIDE the backbone (drm2/drm3 between
    stages, identical to Single224_CASCADE_MSDec_DATin) but WITHOUT the MSConvBlock decoder
    refinement -> isolates the deformable contribution from innovation-1. Single MaxViT bb."""

    def __init__(self, n_class=9, img_size=224, model_scale='small'):
        super().__init__()
        from lib.networks import load_pretrained_weights
        from lib.decoders import CASCADE_Add
        self.conv = nn.Sequential(nn.Conv2d(1, 3, 1), nn.BatchNorm2d(3), nn.ReLU(inplace=True))
        self.backbone = load_pretrained_weights(img_size, model_scale)
        self.channels = [768, 384, 192, 96] if model_scale == 'small' else [512, 256, 128, 64]
        fm = img_size // 32
        self.drm2 = DeformRefine(192, fm * 4, ksize=7)
        self.drm3 = DeformRefine(384, fm * 2, ksize=5)
        self.decoder = CASCADE_Add(channels=self.channels)
        self.out_head1 = nn.Conv2d(self.channels[0], n_class, 1)
        self.out_head2 = nn.Conv2d(self.channels[1], n_class, 1)
        self.out_head3 = nn.Conv2d(self.channels[2], n_class, 1)
        self.out_head4 = nn.Conv2d(self.channels[3], n_class, 1)

    def forward(self, x):
        if x.size(1) == 1:
            x = self.conv(x)
        s = self.backbone.stem(x)
        s = self.backbone.stages[0](s); f0 = s
        s = self.backbone.stages[1](s); f1 = s
        s = self.drm2(s)                                       # deform BEFORE stage2
        s = self.backbone.stages[2](s); f2 = s
        s = self.drm3(s)                                       # deform BEFORE stage3
        s = self.backbone.stages[3](s); f3 = self.backbone.norm(s)
        d4, d3, d2, d1 = self.decoder(f3, [f2, f1, f0])
        p1 = F.interpolate(self.out_head1(d4), scale_factor=32, mode='bilinear', align_corners=False)
        p2 = F.interpolate(self.out_head2(d3), scale_factor=16, mode='bilinear', align_corners=False)
        p3 = F.interpolate(self.out_head3(d2), scale_factor=8, mode='bilinear', align_corners=False)
        p4 = F.interpolate(self.out_head4(d1), scale_factor=4, mode='bilinear', align_corners=False)
        return p1, p2, p3, p4


class Single224_CASCADE_SkipFusion(nn.Module):
    """Innovation-3 candidate: keep CASCADE's AG path and add zero-init selective skip fusion."""

    def __init__(self, n_class=9, img_size=224, model_scale='small'):
        super().__init__()
        from lib.networks import load_pretrained_weights
        from lib.decoders import CASCADE_Add_SkipFusion
        self.conv = nn.Sequential(nn.Conv2d(1, 3, 1), nn.BatchNorm2d(3), nn.ReLU(inplace=True))
        self.backbone = load_pretrained_weights(img_size, model_scale)
        self.channels = [768, 384, 192, 96] if model_scale == 'small' else [512, 256, 128, 64]
        self.decoder = CASCADE_Add_SkipFusion(channels=self.channels)
        self.out_head1 = nn.Conv2d(self.channels[0], n_class, 1)
        self.out_head2 = nn.Conv2d(self.channels[1], n_class, 1)
        self.out_head3 = nn.Conv2d(self.channels[2], n_class, 1)
        self.out_head4 = nn.Conv2d(self.channels[3], n_class, 1)

    def forward(self, x):
        if x.size(1) == 1:
            x = self.conv(x)
        f = list(self.backbone(x))
        d4, d3, d2, d1 = self.decoder(f[3], [f[2], f[1], f[0]])
        p1 = F.interpolate(self.out_head1(d4), scale_factor=32, mode='bilinear', align_corners=False)
        p2 = F.interpolate(self.out_head2(d3), scale_factor=16, mode='bilinear', align_corners=False)
        p3 = F.interpolate(self.out_head3(d2), scale_factor=8, mode='bilinear', align_corners=False)
        p4 = F.interpolate(self.out_head4(d1), scale_factor=4, mode='bilinear', align_corners=False)
        return p1, p2, p3, p4


class Single224_CASCADE_SKDec_SkipFusion(Single224_CASCADE_MSDec_SK):
    """Innovation 1 + 3: SK decoder output refinement plus selective skip fusion."""

    def __init__(self, n_class=9, img_size=224, model_scale='small'):
        super().__init__(n_class=n_class, img_size=img_size, model_scale=model_scale)
        from lib.decoders import CASCADE_Add_SkipFusion
        self.decoder = CASCADE_Add_SkipFusion(channels=self.channels)


class Single224_CASCADE_DATin_SkipFusion(Single224_CASCADE_DATin_NoMS):
    """Innovation 2 + 3: DATin encoder injection plus selective skip fusion."""

    def __init__(self, n_class=9, img_size=224, model_scale='small'):
        super().__init__(n_class=n_class, img_size=img_size, model_scale=model_scale)
        from lib.decoders import CASCADE_Add_SkipFusion
        self.decoder = CASCADE_Add_SkipFusion(channels=self.channels)


class Single224_CASCADE_SKDec_DATin_SkipFusion(Single224_CASCADE_SKDec_DATin):
    """Innovation 1 + 2 + 3: keep the A+B decoder path and add C as a zero-init side adapter.

    This is deliberately more conservative than CASCADE_Add_SkipFusion: all original CASCADE_Add
    module names are preserved, so a trained sk_datin checkpoint can be loaded with strict=False
    and only the C adapters start from scratch.
    """

    def __init__(self, n_class=9, img_size=224, model_scale='small'):
        super().__init__(n_class=n_class, img_size=img_size, model_scale=model_scale)
        from lib.decoders import CASCADE_Add_SkipResidual
        self.decoder = CASCADE_Add_SkipResidual(channels=self.channels)


class Single224_CASCADE_SKDec_DATin_SkipFusionC1(Single224_CASCADE_SKDec_DATin):
    """Innovation 1 + 2 + 3 conservative variant: add C only on the highest-resolution skip."""

    def __init__(self, n_class=9, img_size=224, model_scale='small'):
        super().__init__(n_class=n_class, img_size=img_size, model_scale=model_scale)
        from lib.decoders import CASCADE_Add_SkipResidualC1
        self.decoder = CASCADE_Add_SkipResidualC1(channels=self.channels)


class Single224_CASCADE_SKDec_DATin_JointChannel(Single224_CASCADE_SKDec_DATin):
    """A+B+C with spatial AG and identity-initialized joint channel recalibration."""

    def __init__(self, n_class=9, img_size=224, model_scale='small'):
        super().__init__(n_class=n_class, img_size=img_size, model_scale=model_scale)
        from lib.decoders import CASCADE_Add_JointChannel
        self.decoder = CASCADE_Add_JointChannel(self.decoder, channels=self.channels)


class Single224_CASCADE_SKDec_DATin_JointChannelC1(Single224_CASCADE_SKDec_DATin):
    """A+B+C with identity-initialized channel recalibration at the finest skip."""

    def __init__(self, n_class=9, img_size=224, model_scale='small'):
        super().__init__(n_class=n_class, img_size=img_size, model_scale=model_scale)
        from lib.decoders import CASCADE_Add_JointChannelC1
        self.decoder = CASCADE_Add_JointChannelC1(self.decoder, channels=self.channels)


class Single224_CASCADE_SKDec_DATin_JointChannelC23(Single224_CASCADE_SKDec_DATin):
    """A+B+C with joint channel recalibration on semantic skips C2 and C3."""

    def __init__(self, n_class=9, img_size=224, model_scale='small'):
        super().__init__(n_class=n_class, img_size=img_size, model_scale=model_scale)
        from lib.decoders import CASCADE_Add_JointChannelC23
        self.decoder = CASCADE_Add_JointChannelC23(self.decoder, channels=self.channels)


class Single224_CASCADE_JointChannelC23(MaxViT_CASCADE):
    """C only: final C23 skip fusion on the unchanged Base model."""

    def __init__(self, n_class=9, img_size=224, model_scale='small'):
        super().__init__(n_class=n_class, img_size=img_size, model_scale=model_scale)
        from lib.decoders import CASCADE_Add_JointChannelC23
        self.decoder = CASCADE_Add_JointChannelC23(self.decoder, channels=self.channels)


class Single224_CASCADE_SKDec_JointChannelC23(Single224_CASCADE_MSDec_SK):
    """A+C: final C23 skip fusion on the unchanged SK decoder model."""

    def __init__(self, n_class=9, img_size=224, model_scale='small'):
        super().__init__(n_class=n_class, img_size=img_size, model_scale=model_scale)
        from lib.decoders import CASCADE_Add_JointChannelC23
        self.decoder = CASCADE_Add_JointChannelC23(self.decoder, channels=self.channels)


class Single224_CASCADE_DATin_JointChannelC23(Single224_CASCADE_DATin_NoMS):
    """B+C: final C23 skip fusion on the unchanged DATin encoder model."""

    def __init__(self, n_class=9, img_size=224, model_scale='small'):
        super().__init__(n_class=n_class, img_size=img_size, model_scale=model_scale)
        from lib.decoders import CASCADE_Add_JointChannelC23
        self.decoder = CASCADE_Add_JointChannelC23(self.decoder, channels=self.channels)


class Single224_CASCADE_JointChannelC23Weak(MaxViT_CASCADE):
    """Weaker C: C23 channel modulation bounded to +/-0.25."""

    def __init__(self, n_class=9, img_size=224, model_scale='small'):
        super().__init__(n_class=n_class, img_size=img_size, model_scale=model_scale)
        from lib.decoders import CASCADE_Add_JointChannelC23
        self.decoder = CASCADE_Add_JointChannelC23(
            self.decoder, channels=self.channels, max_delta=0.25)


class Single224_CASCADE_SKDec_JointChannelC23Weak(Single224_CASCADE_MSDec_SK):
    """A plus the same weaker C23 used by the standalone C-weak model."""

    def __init__(self, n_class=9, img_size=224, model_scale='small'):
        super().__init__(n_class=n_class, img_size=img_size, model_scale=model_scale)
        from lib.decoders import CASCADE_Add_JointChannelC23
        self.decoder = CASCADE_Add_JointChannelC23(
            self.decoder, channels=self.channels, max_delta=0.25)


class Single224_CASCADE_DATin_JointChannelC23Weak(Single224_CASCADE_DATin_NoMS):
    """B plus the same weaker C23 used by the standalone C-weak model."""

    def __init__(self, n_class=9, img_size=224, model_scale='small'):
        super().__init__(n_class=n_class, img_size=img_size, model_scale=model_scale)
        from lib.decoders import CASCADE_Add_JointChannelC23
        self.decoder = CASCADE_Add_JointChannelC23(
            self.decoder, channels=self.channels, max_delta=0.25)


class Single224_CASCADE_SKDec_JointChannelC3(Single224_CASCADE_MSDec_SK):
    """Tuned A+C: keep C only on the deepest semantic skip."""

    def __init__(self, n_class=9, img_size=224, model_scale='small'):
        super().__init__(n_class=n_class, img_size=img_size, model_scale=model_scale)
        from lib.decoders import CASCADE_Add_JointChannelC3
        self.decoder = CASCADE_Add_JointChannelC3(self.decoder, channels=self.channels)


class Single224_CASCADE_SKDec_JointChannelC2(Single224_CASCADE_MSDec_SK):
    """A+C with C restricted to the middle semantic skip."""

    def __init__(self, n_class=9, img_size=224, model_scale='small', max_delta=0.5):
        super().__init__(n_class=n_class, img_size=img_size, model_scale=model_scale)
        from lib.decoders import CASCADE_Add_JointChannelC2
        self.decoder = CASCADE_Add_JointChannelC2(
            self.decoder, channels=self.channels, max_delta=max_delta)


class Single224_CASCADE_DATin_JointChannelC2(Single224_CASCADE_DATin_NoMS):
    """B+C with C restricted to the middle semantic skip."""

    def __init__(self, n_class=9, img_size=224, model_scale='small', max_delta=0.25):
        super().__init__(n_class=n_class, img_size=img_size, model_scale=model_scale)
        from lib.decoders import CASCADE_Add_JointChannelC2
        self.decoder = CASCADE_Add_JointChannelC2(
            self.decoder, channels=self.channels, max_delta=max_delta)


class Single224_CASCADE_DATin_ComplementaryC1(Single224_CASCADE_DATin_NoMS):
    """Tuned B+C: use boundary-aware C only on the finest skip."""

    def __init__(self, n_class=9, img_size=224, model_scale='small'):
        super().__init__(n_class=n_class, img_size=img_size, model_scale=model_scale)
        from lib.decoders import CASCADE_Add_ComplementaryC1
        self.decoder = CASCADE_Add_ComplementaryC1(self.decoder, channels=self.channels)


class Single224_CASCADE_DATin_WeakSK_ComplementaryC1(
        Single224_CASCADE_DATin_ComplementaryC1):
    """B+C path plus a bounded A side branch, without changing the B+C RNG stream."""

    def __init__(self, n_class=9, img_size=224, model_scale='small', a_scale=0.1):
        super().__init__(n_class=n_class, img_size=img_size, model_scale=model_scale)
        with torch.random.fork_rng(devices=[]):
            self.ms = nn.ModuleList([MSConvBlockSK(c) for c in self.channels])
        self.a_scale = a_scale

    def forward(self, x):
        if x.size(1) == 1:
            x = self.conv(x)
        s = self.backbone.stem(x)
        s = self.backbone.stages[0](s); f0 = s
        s = self.backbone.stages[1](s); f1 = s
        s = self.drm2(s)
        s = self.backbone.stages[2](s); f2 = s
        s = self.drm3(s)
        s = self.backbone.stages[3](s); f3 = self.backbone.norm(s)
        features = list(self.decoder(f3, [f2, f1, f0]))
        for index, feature in enumerate(features):
            refined = self.ms[index](feature)
            features[index] = feature + self.a_scale * (refined - feature)
        d4, d3, d2, d1 = features
        p1 = F.interpolate(self.out_head1(d4), scale_factor=32, mode='bilinear', align_corners=False)
        p2 = F.interpolate(self.out_head2(d3), scale_factor=16, mode='bilinear', align_corners=False)
        p3 = F.interpolate(self.out_head3(d2), scale_factor=8, mode='bilinear', align_corners=False)
        p4 = F.interpolate(self.out_head4(d1), scale_factor=4, mode='bilinear', align_corners=False)
        return p1, p2, p3, p4


class Single224_CASCADE_DATin_CoarseSK_ComplementaryC1(
        Single224_CASCADE_DATin_ComplementaryC1):
    """B+C path with A restricted to the coarse decoder output."""

    def __init__(self, n_class=9, img_size=224, model_scale='small', a_scale=1.0):
        super().__init__(n_class=n_class, img_size=img_size, model_scale=model_scale)
        with torch.random.fork_rng(devices=[]):
            self.ms_coarse = MSConvBlockSK(self.channels[0])
        self.a_scale = a_scale

    def forward(self, x):
        if x.size(1) == 1:
            x = self.conv(x)
        s = self.backbone.stem(x)
        s = self.backbone.stages[0](s); f0 = s
        s = self.backbone.stages[1](s); f1 = s
        s = self.drm2(s)
        s = self.backbone.stages[2](s); f2 = s
        s = self.drm3(s)
        s = self.backbone.stages[3](s); f3 = self.backbone.norm(s)
        d4, d3, d2, d1 = self.decoder(f3, [f2, f1, f0])
        d4_refined = self.ms_coarse(d4)
        d4 = d4 + self.a_scale * (d4_refined - d4)
        p1 = F.interpolate(self.out_head1(d4), scale_factor=32, mode='bilinear', align_corners=False)
        p2 = F.interpolate(self.out_head2(d3), scale_factor=16, mode='bilinear', align_corners=False)
        p3 = F.interpolate(self.out_head3(d2), scale_factor=8, mode='bilinear', align_corners=False)
        p4 = F.interpolate(self.out_head4(d1), scale_factor=4, mode='bilinear', align_corners=False)
        return p1, p2, p3, p4


class Single224_CASCADE_SKDec_ComplementaryC12(Single224_CASCADE_MSDec_SK):
    """Tuned A+C: preserve A semantics and correct skip disagreement at /8 and /4."""

    def __init__(self, n_class=9, img_size=224, model_scale='small'):
        super().__init__(n_class=n_class, img_size=img_size, model_scale=model_scale)
        from lib.decoders import CASCADE_Add_ComplementaryC12
        self.decoder = CASCADE_Add_ComplementaryC12(
            self.decoder, channels=self.channels)


class Single224_CASCADE_SKDec_DATin_JointChannelC3(Single224_CASCADE_SKDec_DATin):
    """A+B+C with joint channel recalibration only on the deepest semantic skip."""

    def __init__(self, n_class=9, img_size=224, model_scale='small'):
        super().__init__(n_class=n_class, img_size=img_size, model_scale=model_scale)
        from lib.decoders import CASCADE_Add_JointChannelC3
        self.decoder = CASCADE_Add_JointChannelC3(self.decoder, channels=self.channels)


class Single224_CASCADE_SKDec_DATin_ComplementaryC1(Single224_CASCADE_SKDec_DATin):
    """A+B+C with identity-initialized boundary-aware fusion at the finest skip."""

    def __init__(self, n_class=9, img_size=224, model_scale='small'):
        super().__init__(n_class=n_class, img_size=img_size, model_scale=model_scale)
        from lib.decoders import CASCADE_Add_ComplementaryC1
        self.decoder = CASCADE_Add_ComplementaryC1(self.decoder, channels=self.channels)


class Single224_CASCADE_SKDec_DATin_DecoupledC1(
        Single224_CASCADE_SKDec_DATin_ComplementaryC1):
    """ABC-C1 with full coarse A and bounded fine-scale A residuals."""

    def __init__(self, n_class=9, img_size=224, model_scale='small', fine_scale=0.25):
        super().__init__(n_class=n_class, img_size=img_size, model_scale=model_scale)
        self.fine_scale = fine_scale

    def forward(self, x):
        if x.size(1) == 1:
            x = self.conv(x)
        s = self.backbone.stem(x)
        s = self.backbone.stages[0](s); f0 = s
        s = self.backbone.stages[1](s); f1 = s
        s = self.drm2(s)
        s = self.backbone.stages[2](s); f2 = s
        s = self.drm3(s)
        s = self.backbone.stages[3](s); f3 = self.backbone.norm(s)
        d4, d3, d2, d1 = self.decoder(f3, [f2, f1, f0])

        d4, d3 = self.ms[0](d4), self.ms[1](d3)
        d2_refined, d1_refined = self.ms[2](d2), self.ms[3](d1)
        d2 = d2 + self.fine_scale * (d2_refined - d2)
        d1 = d1 + self.fine_scale * (d1_refined - d1)

        p1 = F.interpolate(self.out_head1(d4), scale_factor=32, mode='bilinear', align_corners=False)
        p2 = F.interpolate(self.out_head2(d3), scale_factor=16, mode='bilinear', align_corners=False)
        p3 = F.interpolate(self.out_head3(d2), scale_factor=8, mode='bilinear', align_corners=False)
        p4 = F.interpolate(self.out_head4(d1), scale_factor=4, mode='bilinear', align_corners=False)
        return p1, p2, p3, p4


class Single224_CASCADE_SKDec_DATin_ComplementaryC12(Single224_CASCADE_SKDec_DATin):
    """A+B+C with boundary-aware fusion at the two finest skips."""

    def __init__(self, n_class=9, img_size=224, model_scale='small'):
        super().__init__(n_class=n_class, img_size=img_size, model_scale=model_scale)
        from lib.decoders import CASCADE_Add_ComplementaryC12
        self.decoder = CASCADE_Add_ComplementaryC12(self.decoder, channels=self.channels)


class Single224_CASCADE_SKDec_DATin_HybridC12(Single224_CASCADE_SKDec_DATin):
    """ABC with weak semantic C2 recalibration and complementary C1 boundary fusion."""

    def __init__(self, n_class=9, img_size=224, model_scale='small'):
        super().__init__(n_class=n_class, img_size=img_size, model_scale=model_scale)
        from lib.decoders import CASCADE_Add_HybridC12
        self.decoder = CASCADE_Add_HybridC12(self.decoder, channels=self.channels)


class Single224_CASCADE_MSDec(nn.Module):
    """OUR single MaxViT backbone + CASCADE decoder + multi-scale conv refinement (MSConvBlock)
    on each of the 4 decoder outputs. Single backbone, 224. Best single-backbone model = 83.64."""

    def __init__(self, n_class=9, img_size=224, model_scale='small'):
        super().__init__()
        from lib.networks import load_pretrained_weights
        from lib.decoders import CASCADE_Add
        self.conv = nn.Sequential(nn.Conv2d(1, 3, 1), nn.BatchNorm2d(3), nn.ReLU(inplace=True))
        self.backbone = load_pretrained_weights(img_size, model_scale)
        self.channels = [768, 384, 192, 96] if model_scale == 'small' else [512, 256, 128, 64]
        self.decoder = CASCADE_Add(channels=self.channels)
        self.ms = nn.ModuleList([MSConvBlock(c) for c in self.channels])
        self.out_head1 = nn.Conv2d(self.channels[0], n_class, 1)
        self.out_head2 = nn.Conv2d(self.channels[1], n_class, 1)
        self.out_head3 = nn.Conv2d(self.channels[2], n_class, 1)
        self.out_head4 = nn.Conv2d(self.channels[3], n_class, 1)

    def forward(self, x):
        if x.size(1) == 1:
            x = self.conv(x)
        f = list(self.backbone(x))
        d4, d3, d2, d1 = self.decoder(f[3], [f[2], f[1], f[0]])
        d4, d3, d2, d1 = self.ms[0](d4), self.ms[1](d3), self.ms[2](d2), self.ms[3](d1)
        p1 = F.interpolate(self.out_head1(d4), scale_factor=32, mode='bilinear', align_corners=False)
        p2 = F.interpolate(self.out_head2(d3), scale_factor=16, mode='bilinear', align_corners=False)
        p3 = F.interpolate(self.out_head3(d2), scale_factor=8, mode='bilinear', align_corners=False)
        p4 = F.interpolate(self.out_head4(d1), scale_factor=4, mode='bilinear', align_corners=False)
        return p1, p2, p3, p4


class Single224_DATin_MWSA(Single224_CASCADE_MSDec_DATin):
    """FUSION (standard MaxViT, no DAT++ pretrain): deformable injection (DAT drm2/drm3, datin)
    + dynamic multi-window attention (mwsa, replaces backbone attn) -> two ORTHOGONAL axes:
    deformable=content-adaptive sampling LOCATION, multi-window=content-adaptive receptive SCALE.
    + CASCADE + MSConvBlock + 4-map MUTATION. Our own fusion contribution."""
    def __init__(self, n_class=9, img_size=224, model_scale='small'):
        super().__init__(n_class=n_class, img_size=img_size, model_scale=model_scale)
        from lib.dynamic_attn import DynamicMultiWindowAttention
        n = 0
        for name, module in self.backbone.named_modules():
            if hasattr(module, 'attn') and hasattr(module, 'dim') and hasattr(module, 'num_heads'):
                if not isinstance(module.attn, DynamicMultiWindowAttention):
                    module.attn = DynamicMultiWindowAttention(
                        dim=module.dim, num_heads=module.num_heads, window_sizes=[4, 8, 16])
                    n += 1
        print('FUSION: injected DynamicMultiWindowAttention into %d backbone attn layers (+ DAT drm2/drm3)' % n)
