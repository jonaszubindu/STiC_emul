"""
Publication figure: applying the trained emulator to new observations.

    python docs/figures/make_application_figure.py

writes application_figure.{pdf,svg,png} next to this script, in the style
of pipeline_figure (same colours and box types). Pure matplotlib; the
PDF/SVG are vector and editable.
"""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

HERE = os.path.dirname(os.path.abspath(__file__))
N_ENS = 4

plt.rcParams.update({'font.size': 6.5, 'font.family': 'DejaVu Sans',
                     'mathtext.fontset': 'dejavusans'})
COL = dict(
    obs=('#E3EBF3', '#4A6A8A'),     # observations / data preparation
    stic=('#F6DDBF', '#B5702A'),    # STiC (CPU)
    data=('#EEEEEE', '#666666'),    # data products
    ml=('#CFE0F4', '#2F5F9E'),      # network (GPU)
    val=('#D9EDD5', '#3E7D3A'),     # checks / verification
    out=('#F9E7C7', '#A6761D'),     # outputs
)
LANE_CPU, LANE_GPU = '#FCF4EA', '#EEF4FB'

fig = plt.figure(figsize=(7.2, 5.6))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 72)
ax.set_ylim(0, 56)
ax.set_aspect('equal')
ax.axis('off')


def box(x0, y0, w, h, title, body='', kind='data', fs=6.1, tfs=6.8,
        title_gap=1.3):
    fc, ec = COL[kind]
    ax.add_patch(FancyBboxPatch((x0, y0), w, h,
                                boxstyle='round,pad=0,rounding_size=0.7',
                                fc=fc, ec=ec, lw=0.8, zorder=2))
    ax.text(x0 + w / 2, y0 + h - title_gap, title, ha='center', va='center',
            fontsize=tfs, weight='bold', zorder=3, linespacing=1.2)
    ax.text(x0 + w / 2, y0 + (h - title_gap - 0.6) / 2, body, ha='center',
            va='center', fontsize=fs, zorder=3, linespacing=1.25)


def arrow(p0, p1, text='', color='#333333', ls='-', rad=0.0, tx=None,
          fs=6.0):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle='-|>', mutation_scale=7,
                                 lw=0.8, color=color, ls=ls, zorder=4,
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


ax.text(0.6, 54.6, 'Applying the trained emulator to new observations',
        fontsize=8, va='center')

# lanes
ax.add_patch(FancyBboxPatch((32.0, 34.2), 39.4, 18.0,
                            boxstyle='round,pad=0,rounding_size=1',
                            fc=LANE_GPU, ec='none', zorder=0))
ax.text(51.7, 51.0, 'GPU: trained network, no inversion', ha='center',
        fontsize=6.5, color=COL['ml'][1], weight='bold')
ax.add_patch(FancyBboxPatch((14.8, 1.0), 43.0, 29.8,
                            boxstyle='round,pad=0,rounding_size=1',
                            fc=LANE_CPU, ec='none', zorder=0))
ax.text(36.3, 29.6, 'CPU cluster: STiC (optional)', ha='center',
        fontsize=6.5, color=COL['stic'][1], weight='bold')

# ---- row 1: data -> check -> network -> atmosphere ------------------------ #
box(0.6, 36.0, 13.4, 13.0, 'New observations',
    'CHROMIS Ca II K, 4000 Å\nCRISP Fe I 6302 Å,\nCa II 8542 Å\n\n'
    'other FOVs, frames of\na time series', kind='obs')
box(16.2, 36.0, 14.0, 13.0, 'Data preparation',
    'as for a coupled STiC\ninversion: alignment,\n'
    r'destretch $\mathbf{ds}$, PSFs,' + '\nrebin geometry,\n'
    'instrumental profiles', kind='obs')
box(33.4, 36.0, 11.6, 13.0, 'Input check',
    'regions, wavelengths\nand Stokes as in\ntraining (else stop)\n\n'
    'intensity levels in\nunits of the training\nspread (flag)', kind='val',
    fs=5.9)
box(46.6, 36.0, 11.0, 13.0, 'Trained\nensemble',
    f'{N_ENS} networks,\nFig. 1b\n\none forward pass\nper pixel',
    kind='ml', title_gap=2.2)
box(59.2, 36.0, 12.2, 13.0, 'Emulated\natmosphere',
    'values at the STiC\nnodes → stratifications\n'
    + r'in $\log\tau_{500}$:' + '\n'
    + r'$T,\ v_\mathrm{los},\ v_\mathrm{turb},\ B_\parallel,\ B_\perp,\ \phi$'
    + '\n' + r'$+\ \sigma_\mathrm{epistemic},\ \sigma_\mathrm{aleatoric}$',
    kind='out', title_gap=2.2, fs=5.9)

arrow((14.0, 42.5), (16.2, 42.5))
arrow((30.2, 42.5), (33.4, 42.5))
arrow((45.0, 42.5), (46.6, 42.5))
arrow((57.6, 42.5), (59.2, 42.5))
ax.text(23.2, 34.6, 'obs_*.nc, inst_*.nc, input.cfg', fontsize=5.8,
        ha='center', style='italic', color='#4A6A8A')

# ---- row 2: uses of the emulated atmosphere ------------------------------- #
box(59.2, 17.0, 12.2, 12.0, 'Direct use',
    'maps and height\nprofiles with\nuncertainties;\ncheckerboard\n'
    '(Nyquist) check', kind='out')
box(45.2, 17.0, 11.2, 12.0, 'STiC model',
    'emulated atmosphere\n+ top gas pressure\nof the starting model\n'
    '(hydrostatic\nequilibrium by STiC)', kind='data', fs=5.9)
box(31.0, 17.0, 11.6, 12.0, 'Synthesis',
    'non-LTE spectra\n(STiC mode 2),\ninstrumental profiles,\n'
    r'$D_r$' + ' to each\ninstrument grid', kind='stic', fs=5.9)
box(16.2, 17.0, 12.2, 12.0, r'$\chi^2$ verification',
    'emulated vs. observed\nprofiles per region\nand pixel; vs. the\n'
    r'inversion $\chi^2$' + '\nwhere available', kind='val', fs=5.9)
box(31.0, 2.4, 11.6, 11.6, 'Coupled STiC',
    'emulated atmosphere\nas initial model:\nfew iterations\n'
    'instead of many\ncycles', kind='stic', fs=5.9)
box(16.2, 2.4, 12.2, 11.6, 'New labels',
    r'where $\sigma$ or $\chi^2$ is' + '\nhigh: tile inversions\n'
    '(refined map as\ninitial model)', kind='data', fs=5.9)

arrow((65.3, 36.0), (65.3, 29.0))                         # atmos -> direct use
arrow((61.0, 36.0), (54.5, 29.0))                         # atmos -> STiC model
arrow((45.2, 23.0), (42.6, 23.0))                         # model -> synthesis
arrow((31.0, 23.0), (28.4, 23.0))                         # synthesis -> chi2
arrow((47.5, 17.0), (42.6, 10.5), rad=-0.15)              # model -> coupled STiC
arrow((31.0, 8.2), (28.4, 8.2))                           # refined -> labels
arrow((22.3, 17.0), (22.3, 14.0), text='high ' + r'$\chi^2$',
      tx=(22.3, 15.5), fs=5.6)

# active learning: new labels -> retraining
polyline([(16.2, 8.2), (8.0, 8.2), (8.0, 32.4), (52.1, 32.4), (52.1, 36.0)],
         text='retraining with the new labels (active learning)',
         color=COL['out'][1], ls='--', tx=(25.0, 32.4))

for ext in ('pdf', 'svg', 'png'):
    p = os.path.join(HERE, f'application_figure.{ext}')
    fig.savefig(p, dpi=300 if ext == 'png' else None)
    print('wrote', p)
