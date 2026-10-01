"""
Publication figure: the STiC_emul pipeline (a) and the network (b).

    python docs/figures/make_pipeline_figure.py

writes pipeline_figure.{pdf,svg,png} next to this script. Pure
matplotlib; the PDF/SVG are vector and editable in Inkscape/Illustrator.
Dataset-specific numbers (grids, channels) are set in REGIONS below.
"""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch

HERE = os.path.dirname(os.path.abspath(__file__))

# ---- dataset-specific numbers (inv_fovb data set) ------------------------- #
# label, wavelength, grid (ny, nx), cell size [reference px], input channels
REGIONS = [
    ('CHROMIS', r'Ca II K', (140, 140), 1.00, 31),
    ('CHROMIS', r'4000 $\AA$ cont.', (140, 140), 1.00, 1),
    ('CRISP', r'Fe I 6302 $\AA$', (73, 72), 1.89, 48),
    ('CRISP', r'Ca II 8542 $\AA$', (53, 53), 2.57, 64),
]

# ---- network / training settings (emulator/emu_config.py) ---------------- #
C_FEAT, ENC_LAYERS, DEC_HIDDEN, N_ENS = 64, 3, 512, 4
N_NODES, N_QUANT = 16, 7

# ---- style ---------------------------------------------------------------- #
plt.rcParams.update({'font.size': 6.5, 'font.family': 'DejaVu Sans',
                     'mathtext.fontset': 'dejavusans'})
COL = dict(
    obs=('#E3EBF3', '#4A6A8A'),     # observations / operators
    stic=('#F6DDBF', '#B5702A'),    # coupled STiC (CPU)
    data=('#EEEEEE', '#666666'),    # data products
    ml=('#CFE0F4', '#2F5F9E'),      # network / training (GPU)
    geo=('#E5D8F1', '#6B4C9A'),     # geometry-aware sampling
    val=('#D9EDD5', '#3E7D3A'),     # validation
    out=('#F9E7C7', '#A6761D'),     # outputs
)
LANE_CPU, LANE_GPU = '#FCF4EA', '#EEF4FB'

fig = plt.figure(figsize=(7.2, 10.0))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 72)
ax.set_ylim(0, 100)
ax.set_aspect('equal')
ax.axis('off')


def box(x0, y0, w, h, title, body='', kind='data', fs=6.3, tfs=6.8,
        ls='-', title_gap=1.3):
    fc, ec = COL[kind]
    ax.add_patch(FancyBboxPatch((x0, y0), w, h,
                                boxstyle='round,pad=0,rounding_size=0.7',
                                fc=fc, ec=ec, lw=0.8, ls=ls, zorder=2))
    if body:
        ax.text(x0 + w / 2, y0 + h - title_gap, title, ha='center',
                va='center', fontsize=tfs, weight='bold', zorder=3)
        ax.text(x0 + w / 2, y0 + (h - title_gap - 0.6) / 2, body,
                ha='center', va='center', fontsize=fs, zorder=3,
                linespacing=1.25)
    else:
        ax.text(x0 + w / 2, y0 + h / 2, title, ha='center', va='center',
                fontsize=tfs, weight='bold', zorder=3, linespacing=1.25)


def arrow(p0, p1, text='', color='#333333', ls='-', rad=0.0, tx=None,
          fs=6.0, lw=0.8):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle='-|>', mutation_scale=7,
                                 lw=lw, color=color, ls=ls, zorder=4,
                                 connectionstyle=f'arc3,rad={rad}',
                                 shrinkA=0, shrinkB=0))
    if text:
        x, y = tx if tx else ((p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2)
        ax.text(x, y, text, fontsize=fs, ha='center', va='center',
                color=color, style='italic', zorder=5,
                bbox=dict(fc='white', ec='none', pad=0.3, alpha=0.9))


def polyline(points, text='', color='#333333', ls='-', tx=None, fs=6.0):
    xs, ys = zip(*points[:-1])
    ax.plot(xs, ys, color=color, lw=0.8, ls=ls, zorder=4,
            solid_capstyle='butt')
    arrow(points[-2], points[-1], color=color, ls=ls)
    if text:
        ax.text(*tx, text, fontsize=fs, ha='center', va='center',
                color=color, style='italic', zorder=5,
                bbox=dict(fc='white', ec='none', pad=0.3, alpha=0.9))


def grid_icon(x0, y0, size, n, color='#4A6A8A', fc='white'):
    """Square field of view with n x n pixels (resolution sketch)."""
    ax.add_patch(Rectangle((x0, y0), size, size, fc=fc, ec=color, lw=0.7,
                           zorder=3))
    for i in range(1, n):
        t = size * i / n
        ax.plot([x0 + t, x0 + t], [y0, y0 + size], color=color, lw=0.3,
                alpha=0.6, zorder=3)
        ax.plot([x0, x0 + size], [y0 + t, y0 + t], color=color, lw=0.3,
                alpha=0.6, zorder=3)


# =========================================================================== #
# (a) pipeline
# =========================================================================== #
ax.text(0.6, 98.8, '(a)', fontsize=9, weight='bold', va='center')
ax.text(3.0, 98.8, 'Label generation, training and the bootstrap loop',
        fontsize=8, va='center')

# lanes
ax.add_patch(FancyBboxPatch((15.6, 54.0), 29.6, 42.6,
                            boxstyle='round,pad=0,rounding_size=1',
                            fc=LANE_CPU, ec='none', zorder=0))
ax.add_patch(FancyBboxPatch((46.0, 54.0), 25.4, 42.6,
                            boxstyle='round,pad=0,rounding_size=1',
                            fc=LANE_GPU, ec='none', zorder=0))
ax.text(30.4, 95.3, 'CPU cluster: coupled STiC (MPI, SLURM)', ha='center',
        fontsize=6.5, color=COL['stic'][1], weight='bold')
ax.text(58.7, 95.3, 'GPU cluster: PyTorch', ha='center', fontsize=6.5,
        color=COL['ml'][1], weight='bold')

# inputs
box(0.6, 80.0, 13.8, 13.4, 'Observations',
    'CHROMIS Ca II K, 4000 Å\n(reference grid)\nCRISP Fe I 6302 Å,\n'
    'Ca II 8542 Å\n(coarser grids)', kind='obs')
box(0.6, 61.0, 13.8, 15.4, 'Spatial operators',
    r'$D_r = \mathrm{LT}_r \cdot \mathrm{PSF}$' + '\nper spectral region:\n'
    'destretch + rebin\nto each instrument grid', kind='obs')

# CPU lane
box(17.0, 82.0, 12.4, 11.4, 'Coupled STiC',
    'non-LTE inversion\nof the full FOV,\nspatially coupled\nthrough ' +
    r'$D_r$', kind='stic')
box(17.0, 63.0, 12.4, 13.4, 'Tile selection',
    'k-means on tile\ncontent, stratified\ndraw; 56×56 px\ntiles, 8 px apron,\n'
    r'$D_r$' + ' re-based exactly', kind='stic')
box(32.8, 63.0, 11.4, 13.4, 'Tile inversions',
    'coupled STiC,\none SLURM array\njob per tile', kind='stic')
box(32.8, 82.0, 11.4, 11.4, 'Label records',
    'stratifications\n+ observed windows\nat native resolution\n'
    r'+ per-pixel $\chi^2$', kind='data')

arrow((14.4, 87.0), (17.0, 87.0))                              # obs -> STiC
arrow((14.4, 72.5), (17.0, 85.0), rad=0.25)                    # ops -> STiC
arrow((14.4, 68.0), (17.0, 68.0))                              # ops -> tiles
arrow((29.4, 87.7), (32.8, 87.7))                              # STiC -> labels
ax.text(31.1, 89.6, 'harvest', fontsize=6.0, ha='center', style='italic')
arrow((23.2, 82.0), (23.2, 76.4), text='initial\nmodel', tx=(23.2, 79.2))
arrow((29.4, 69.7), (32.8, 69.7))                              # tiles -> inv
arrow((38.5, 76.4), (38.5, 82.0), text='harvest', tx=(38.5, 79.2))

# GPU lane
box(47.4, 82.0, 11.0, 11.4, 'Label selection',
    r'pixel $\chi^2$ filter' + '\n(best 50 %)\nspatial hold-out\n'
    '(15 % of 20 px\nblocks, 3 px guard)', kind='ml', fs=6.1)
box(47.4, 63.0, 11.0, 13.4, 'Ensemble\ntraining',
    f'{N_ENS} networks, panel (b)\nGaussian NLL,\nearly stopping on\n'
    'held-out pixels', kind='ml', title_gap=2.4, fs=6.1)
box(60.0, 63.0, 11.0, 13.4, 'Prediction',
    'full-FOV\natmosphere\n' + r'+ $\sigma_\mathrm{aleatoric}$,'
    + '\n' + r'$\sigma_\mathrm{epistemic}$', kind='out')
box(60.0, 82.0, 11.0, 11.4, 'Validation',
    'vs. inversion:\nmaps, per-pixel\nscatter, height\nprofiles', kind='val')

arrow((44.2, 87.7), (47.4, 87.7))
ax.text(45.8, 89.6, 'rsync', fontsize=6.0, ha='center', style='italic')
arrow((52.9, 82.0), (52.9, 76.4))
arrow((58.4, 69.7), (60.0, 69.7))
arrow((65.5, 76.4), (65.5, 82.0))

# bootstrap loop: prediction initializes the next tile round
polyline([(65.5, 63.0), (65.5, 57.2), (38.5, 57.2), (38.5, 63.0)],
         text='bootstrap: the prediction initializes the next tile round',
         color=COL['out'][1], tx=(51.8, 57.2))

# =========================================================================== #
# (b) network
# =========================================================================== #
ax.text(0.6, 50.6, '(b)', fontsize=9, weight='bold', va='center')
ax.text(3.0, 50.6, 'Multi-resolution emulator network (one ensemble member)',
        fontsize=8, va='center')

row_y = [42.2, 33.6, 25.0, 16.4]       # bottom of each region row
ICON = 6.4
fx_feat = 33.2                          # x of feature-map icons
geo_x0, geo_x1 = 41.6, 53.0
for (inst, line, (ny, nx), cell, nch), y0 in zip(REGIONS, row_y):
    # input grid icon (same FOV, different pixel density)
    n_icon = max(3, round(14 / cell))
    grid_icon(0.8, y0, ICON, n_icon)
    ax.text(7.8, y0 + ICON - 0.8, f'{inst}  {line}', fontsize=6.3,
            weight='bold', va='center')
    ax.text(7.8, y0 + ICON / 2 - 0.2,
            f'{ny}×{nx} px, cell {cell:.2f}\n{nch} λ×Stokes channels\n'
            '+ pixel-weight mask', fontsize=6.0, va='center',
            linespacing=1.2)
    # query position marker on the icon
    ax.plot(0.8 + 0.62 * ICON, y0 + 0.55 * ICON, 'o', ms=2.4,
            color='#C0392B', zorder=5)
    # encoder
    box(19.4, y0 + 0.4, 11.6, ICON - 0.8,
        f'Encoder $E_r$',
        f'{ENC_LAYERS} × [conv 3×3, {C_FEAT}, GELU]\nnative resolution',
        kind='ml', fs=6.0, tfs=6.3, title_gap=1.15)
    arrow((17.6, y0 + ICON / 2), (19.4, y0 + ICON / 2))
    # feature map stack
    for k in range(3):
        ax.add_patch(Rectangle((fx_feat + 0.5 * k, y0 + 0.6 + 0.5 * k),
                               4.2, 4.2, fc=COL['ml'][0], ec=COL['ml'][1],
                               lw=0.6, zorder=3 + k))
    ax.text(fx_feat + 2.6, y0 - 0.15, r'$F_r$: ' + f'{C_FEAT}' + r'$\times H_r\times W_r$',
            fontsize=6.0, ha='center', va='center')
    arrow((31.0, y0 + ICON / 2), (fx_feat, y0 + ICON / 2))
    # into the geometry-aware sampler
    arrow((fx_feat + 5.2, y0 + ICON / 2), (geo_x0, y0 + ICON / 2),
          color=COL['geo'][1])

# legend for the query marker
ax.plot(1.4, 14.0, 'o', ms=2.4, color='#C0392B')
ax.text(2.4, 14.0, r'query pixel $\mathbf{k}$: the same sky position, '
        'looked up on every instrument grid', fontsize=6.0, va='center')

# geometry-aware sampling (spans all rows)
y_lo, y_hi = row_y[-1] - 0.2, row_y[0] + ICON + 0.2
fc, ec = COL['geo']
ax.add_patch(FancyBboxPatch((geo_x0, y_lo), geo_x1 - geo_x0, y_hi - y_lo,
                            boxstyle='round,pad=0,rounding_size=0.7',
                            fc=fc, ec=ec, lw=0.8, zorder=2))
gx = (geo_x0 + geo_x1) / 2
ax.text(gx, y_hi - 1.6, 'Geometry-aware\nsampling', ha='center',
        va='center', fontsize=6.8, weight='bold', linespacing=1.2)
ax.text(gx, (y_lo + y_hi) / 2 + 0.8,
        'query pixel ' + r'$\mathbf{k}$' + '\non the reference grid\n\n'
        'invert destretch\n' + r'$\mathbf{w} \approx \mathbf{k}-\mathbf{ds}(\mathbf{k})$'
        + '\n\nposition in grid ' + r'$r$' + '\nfrom stored rebin\ngeometry\n\n'
        'bilinear ' + r'$f_r = F_r(\mathbf{w})$' + '\n+ sub-cell offset '
        + r'$\delta_r$', ha='center', va='center', fontsize=6.0,
        linespacing=1.22)

# concatenation
cx0 = 54.2
ax.add_patch(Rectangle((cx0, y_lo + 6), 1.6, y_hi - y_lo - 12,
                       fc='#DDDDDD', ec='#666666', lw=0.6, zorder=2))
ax.text(cx0 + 0.8, (y_lo + y_hi) / 2,
        f'concat  {len(REGIONS)} × ({C_FEAT}+2) = {len(REGIONS) * (C_FEAT + 2)}',
        rotation=90, ha='center', va='center', fontsize=6.0)
arrow((geo_x1, (y_lo + y_hi) / 2), (cx0, (y_lo + y_hi) / 2))

# decoder MLP
dx0, dw = 57.0, 8.2
dec_y = [36.0, 29.4, 22.8]
for i, y0 in enumerate(dec_y):
    box(dx0, y0, dw, 5.2, f'Linear {DEC_HIDDEN}',
        'LayerNorm, GELU\nDropout 0.05', kind='ml', fs=6.0, tfs=6.3,
        title_gap=1.15)
    if i:
        arrow((dx0 + dw / 2, dec_y[i - 1]), (dx0 + dw / 2, y0 + 5.2))
arrow((cx0 + 1.6, (y_lo + y_hi) / 2), (dx0, dec_y[0] + 2.6), rad=-0.2)
ax.text(dx0 + dw / 2, 42.4, 'Decoder (per pixel)', ha='center',
        fontsize=6.6, weight='bold')

# heads
hx0, hw = 66.2, 5.4
box(hx0, 28.0, hw, 4.8, r'$\mu$', f'{N_NODES * N_QUANT} values',
    kind='out', fs=6.0, tfs=7.0, title_gap=1.3)
box(hx0, 21.4, hw, 4.8, r'$\log\sigma^2$', f'{N_NODES * N_QUANT} values',
    kind='out', fs=6.0, tfs=7.0, title_gap=1.3)
arrow((dx0 + dw, 25.4), (hx0, 30.4), rad=0.0)
arrow((dx0 + dw, 25.4), (hx0, 23.8), rad=0.0)

# output definition
ax.text(64.2, 20.4,
        f'{N_QUANT} quantities × {N_NODES} nodes in\n' + r'$\log\tau_{500}$'
        + ' = −7 … 0.8:\n' + r'$\log T,\ v_\mathrm{los},\ v_\mathrm{turb},$'
        + '\n' + r'$B_\parallel,\ B_\perp,\ \sin2\phi,\ \cos2\phi$',
        ha='center', va='top', fontsize=6.0, linespacing=1.3)

# training / ensemble strip
ax.add_patch(FancyBboxPatch((0.6, 2.0), 70.8, 9.2,
                            boxstyle='round,pad=0,rounding_size=0.7',
                            fc='#F7F7F7', ec='#999999', lw=0.6, zorder=1))
ax.text(1.6, 9.6, 'Training and uncertainty', fontsize=6.8,
        weight='bold', va='center')
ax.text(1.6, 5.8,
        r'Loss: Gaussian negative log-likelihood $\frac{1}{2}\left[\log\sigma^2'
        r' + (y-\mu)^2/\sigma^2\right]$ on standardized targets, plus '
        r'$\lambda\,\sum(\partial^2\mu/\partial\log\tau^2)^2$ with '
        r'$\lambda=10^{-3}$' + '\n'
        r'AdamW (lr $3\times10^{-4}$, cosine schedule), ≤20 000 steps of '
        r'4 records × ≤2048 pixels, early stopping on the held-out loss'
        + '\n' + f'Ensemble of {N_ENS} members (different initializations): '
        r'prediction = ensemble mean, '
        r'$\sigma_\mathrm{aleatoric} = \langle\sigma^2\rangle^{1/2}$, '
        r'$\sigma_\mathrm{epistemic}$ = spread of the member means',
        fontsize=6.0, va='center', linespacing=1.45)

for ext in ('pdf', 'svg', 'png'):
    p = os.path.join(HERE, f'pipeline_figure.{ext}')
    fig.savefig(p, dpi=300 if ext == 'png' else None)
    print('wrote', p)
