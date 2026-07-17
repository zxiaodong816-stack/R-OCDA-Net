
import math
import torch
import torch.nn as nn


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))

class OCAConv2d(nn.Module):
    """
    Pinwheel-shaped Conv: four directional (1xK)/(Kx1) convolutions with 1x1 fusion.
    act=False for attention logits; s for anisotropic downsampling: (1,s)/(s,1).
    """
    def __init__(self, c1, c2, k=7, s=1, act=False):
        super().__init__()
        assert isinstance(k, int) and k >= 1
        pb = max(1, math.ceil(c2 / 4))
        self.pad = nn.ModuleList([
            nn.ZeroPad2d((k - 1, 0, 0, 0)),  # left
            nn.ZeroPad2d((0, k - 1, 0, 0)),  # right
            nn.ZeroPad2d((0, 0, k - 1, 0)),  # top
            nn.ZeroPad2d((0, 0, 0, k - 1)),  # bottom
        ])
        self.cw = Conv(c1, pb, (1, k), s=(1, s), p=0, act=act)
        self.ch = Conv(c1, pb, (k, 1), s=(s, 1), p=0, act=act)
        self.mix = Conv(pb * 4, c2, 1, s=1, p=0, act=act)

    def forward(self, x):
        y0 = self.cw(self.pad[0](x))
        y1 = self.cw(self.pad[1](x))
        y2 = self.ch(self.pad[2](x))
        y3 = self.ch(self.pad[3](x))
        return self.mix(torch.cat([y0, y1, y2, y3], dim=1))

class APBottleneck(nn.Module):
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        q = max(1, math.ceil(c_ / 4))

        if isinstance(k, (list, tuple)):
            k1 = k[0]
            k2 = k[1]
        else:
            k1 = k2 = k

        if isinstance(k1, int):
            kh1, kw1 = k1, k1
        else:
            assert len(k1) == 2, "k[0] must be int or (kh, kw)"
            kh1, kw1 = int(k1[0]), int(k1[1])

        self.pad_w = nn.ModuleList([
            nn.ZeroPad2d((max(0, kw1 - 1), 0, 0, 0)),
            nn.ZeroPad2d((0, max(0, kw1 - 1), 0, 0)),
        ])
        self.pad_h = nn.ModuleList([
            nn.ZeroPad2d((0, 0, max(0, kh1 - 1), 0)),
            nn.ZeroPad2d((0, 0, 0, max(0, kh1 - 1))),
        ])

        self.cw = Conv(c1, q, (1, max(1, kw1)), 1, p=0, act=True)
        self.ch = Conv(c1, q, (max(1, kh1), 1), 1, p=0, act=True)

        self.cv2 = Conv(4 * q, c2, k2, 1, g=g, act=True)

        self.add = shortcut and c1 == c2

    def forward(self, x):
        y0 = self.cw(self.pad_w[0](x))
        y1 = self.cw(self.pad_w[1](x))
        y2 = self.ch(self.pad_h[0](x))
        y3 = self.ch(self.pad_h[1](x))

        y = torch.cat([y0, y1, y2, y3], dim=1)
        y = self.cv2(y)
        return x + y if self.add else y


class SpatialAttention_CGA(nn.Module):
    def __init__(self, k=7):
        super().__init__()
        self.sa = OCAConv2d(2, 1, k=k, s=1, act=False)

    def forward(self, x):
        x_avg = torch.mean(x, dim=1, keepdim=True)
        x_max, _ = torch.max(x, dim=1, keepdim=True)
        x2 = torch.cat([x_avg, x_max], dim=1)
        sattn = self.sa(x2)
        return sattn


class ChannelAttention_CGA(nn.Module):
    def __init__(self, dim, reduction=8):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        rd = max(1, dim // reduction)
        self.ca = nn.Sequential(
            nn.Conv2d(dim, rd, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(rd, dim, 1, bias=True),
        )

    def forward(self, x):
        x_gap = self.gap(x)
        cattn = self.ca(x_gap)
        return cattn


class PixelAttention_CGA(nn.Module):
    def __init__(self, dim, k=7):
        super().__init__()
        self.pa2 = OCAConv2d(2 * dim, dim, k=k, s=1, act=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, pattn1):
        x2 = torch.cat([x, pattn1], dim=1)
        pattn2 = self.pa2(x2)
        pattn2 = self.sigmoid(pattn2)
        return pattn2


class OCDAFusion(nn.Module):
    def __init__(self, dim, reduction=8, fuse_k=3):
        super().__init__()
        self.sa = SpatialAttention_CGA(k=7)
        self.ca = ChannelAttention_CGA(dim, reduction)
        self.pa = PixelAttention_CGA(dim, k=7)

        self.fuse_pconv = OCAConv2d(2 * dim, dim, k=fuse_k, s=1, act=True)
        self.refine_block = APBottleneck(dim, dim, shortcut=True, g=1, k=(3, 3), e=0.5)

        self.proj = nn.Conv2d(dim, dim, 1, bias=True)

    def forward(self, data):
        x, y = data
        initial = x + y
        initial = initial + self.fuse_pconv(torch.cat([x, y], dim=1))
        initial = self.refine_block(initial)

        cattn = self.ca(initial)
        sattn = self.sa(initial)
        pattn1 = sattn + cattn
        pattn2 = self.pa(initial, pattn1)

        out = initial + pattn2 * x + (1.0 - pattn2) * y
        out = self.proj(out)
        return out
