"""
Multi-resolution emulator network for coupled STiC.

  - one small CNN encoder per spectral region, run at the region's
    native resolution (no resampling of the data, no transposed
    convolutions anywhere — checkerboard hygiene);
  - LT-aware feature sampling: an absolute fine-grid coordinate is
    mapped into each region's image through the *stored* geometry
    (destretch + rebin cell layout), and encoder features are sampled
    bilinearly there;
  - a LIIF-style coordinate decoder (MLP) consumes the concatenated
    multi-region features plus the sub-cell offsets and predicts a
    heteroscedastic stratification (mean + log-variance per channel).

The mapping fine->region for the warped coarse regions inverts
w + ds(w) = k to first order (w ~ k - ds(k)); mis-registration of a small
fraction of a coarse pixel is absorbed by the learned features.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# geometry
# --------------------------------------------------------------------------- #

def region_grid_coords(region, coords):
    """Absolute fine-grid (y, x) -> continuous (row, col) index in the
    region's image (integer index = pixel center), plus sub-cell offsets.

    region: dict with rd, rs, ds, m0 (see nn_dataset).
    coords: (npix, 2) float array of absolute fine (y, x).
    Returns (rc (npix, 2), frac (npix, 2)).
    """
    ky = coords[:, 0].astype('float64')
    kx = coords[:, 1].astype('float64')
    if region['ds'] is not None:
        m0x, m0y = region['m0']
        tny, tnx = region['ds'].shape[1:]
        iy = np.clip(np.round(ky - m0y).astype(int), 0, tny - 1)
        ix = np.clip(np.round(kx - m0x).astype(int), 0, tnx - 1)
        wx = kx - region['ds'][0][iy, ix]
        wy = ky - region['ds'][1][iy, ix]
    else:
        wx, wy = kx, ky
    rdx, rdy = region['rd']
    rsx, rsy = region['rs']
    col = (wx + 0.5 - rsx) / rdx
    row = (wy + 0.5 - rsy) / rdy
    rc = np.stack([row, col], axis=1)
    frac = rc - np.floor(rc) - 0.5     # position within the obs cell
    return rc, frac


def sample_features(feat, rc):
    """Bilinear-sample a (C, H, W) feature map at continuous indices
    rc (npix, 2) (row, col; integer = pixel center). Returns (npix, C)."""
    _, H, W = feat.shape
    gx = 2.0 * (rc[:, 1] + 0.5) / W - 1.0
    gy = 2.0 * (rc[:, 0] + 0.5) / H - 1.0
    grid = torch.stack([gx, gy], dim=1).view(1, -1, 1, 2).to(feat.dtype)
    out = F.grid_sample(feat[None], grid.to(feat.device),
                        mode='bilinear', padding_mode='border',
                        align_corners=False)
    return out[0, :, :, 0].T           # (npix, C)


# --------------------------------------------------------------------------- #
# modules
# --------------------------------------------------------------------------- #

class RegionEncoder(nn.Module):
    def __init__(self, c_in, c_feat, n_layers=3):
        super().__init__()
        layers = []
        c = c_in
        for _ in range(n_layers):
            layers += [nn.Conv2d(c, c_feat, 3, padding=1), nn.GELU()]
            c = c_feat
        self.body = nn.Sequential(*layers)

    def forward(self, img):            # (C_in, H, W) -> (C_feat, H, W)
        return self.body(img[None])[0]


class CoupledEmulator(nn.Module):
    """Encoders per region + coordinate decoder (mean, logvar)."""

    def __init__(self, enc_channels, n_out, c_feat=64, enc_layers=3,
                 dec_hidden=(512, 512, 512), dropout=0.05):
        super().__init__()
        self.tags = sorted(enc_channels)
        self.encoders = nn.ModuleDict({
            t: RegionEncoder(enc_channels[t], c_feat, enc_layers)
            for t in self.tags})
        d_in = len(self.tags) * (c_feat + 2)   # features + frac offsets
        layers = []
        d = d_in
        for h in dec_hidden:
            layers += [nn.Linear(d, h), nn.LayerNorm(h), nn.GELU(),
                       nn.Dropout(dropout)]
            d = h
        self.dec = nn.Sequential(*layers)
        self.head_mu = nn.Linear(d, n_out)
        self.head_lv = nn.Linear(d, n_out)

    def forward(self, sample, coords, device):
        """sample: dict from nn_dataset; coords: (npix, 2) numpy.
        Returns (mu, logvar) tensors (npix, n_out)."""
        parts = []
        for t in self.tags:
            reg = sample['regions'][t]
            img = torch.as_tensor(reg['img'], device=device)
            feat = self.encoders[t](img)
            rc, frac = region_grid_coords(reg, coords)
            f = sample_features(feat, torch.as_tensor(
                rc.astype('float32'), device=device))
            parts.append(f)
            parts.append(torch.as_tensor(frac.astype('float32'),
                                         device=device))
        z = self.dec(torch.cat(parts, dim=1))
        return self.head_mu(z), torch.clamp(self.head_lv(z), -12.0, 6.0)


def loss_fn(mu, logvar, y, depth_groups, smooth_lambda):
    nll = 0.5 * (logvar + (y - mu) ** 2 * torch.exp(-logvar)).mean()
    if smooth_lambda > 0:
        pen = 0.0
        for (i, j) in depth_groups:
            d2 = mu[:, i + 2:j] - 2.0 * mu[:, i + 1:j - 1] + mu[:, i:j - 2]
            pen = pen + (d2 ** 2).mean()
        nll = nll + smooth_lambda * pen / max(len(depth_groups), 1)
    return nll


def make_depth_groups(target_names):
    names = list(target_names)
    groups, i = [], 0
    while i < len(names):
        j = i
        while j < len(names) and names[j] == names[i]:
            j += 1
        if j - i > 2:
            groups.append((i, j))
        i = j
    return groups
