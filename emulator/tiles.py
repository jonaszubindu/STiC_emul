"""
Cut self-contained coupled-STiC run directories for small tiles of the
fine (model) grid — the label-generation unit of the NN emulator.

A tile is a window [mx0:mx0+tnx) x [my0:my0+tny) on the fine grid,
*including* the apron; apron discard happens later at dataset build.
For each spectral region the observed pixels overlapping the tile are
extracted and the stored linear-transformation chain is re-based so that
the tile's operators are consistent with the full-FOV ones:

  - warp (LT 10 destretch part): ds maps are windowed; per-pixel shifts
    are unchanged (they are expressed in pixels of the fine grid).
  - rebin part: input grid origin shifts by (mx0*dx, my0*dy); the output
    grid keeps its cell geometry, restricted to the overlapping output
    pixels, origin shifted accordingly.
  - PSF: unchanged (row renormalization differs near tile borders; that
    error lives in the apron by construction).

Following tools/prepInv.py, coarse-region pixels whose footprint is not
fully covered by the tile get pixel_weight = 0 (coverage = warp+rebin of
a ones-image, threshold 0.9999).
"""

import os
import re
import numpy as np

from . import degradation as dg

LT_NARGS = dg.LT_NARGS


# --------------------------------------------------------------------------- #
# coupled-format profile I/O (pycSTiC coupled_stic.py format)
# --------------------------------------------------------------------------- #

class CoupledObs:
    """Minimal reader/writer for coupled obs files (psf/lts/ltargs/ds)."""

    def __init__(self, filename=None):
        if filename:
            self.read(filename)

    def read(self, filename):
        from netCDF4 import Dataset
        with Dataset(filename, 'r') as f:
            d = f.dimensions
            self.nt = len(d['time']); self.ny = len(d['y'])
            self.nx = len(d['x']); self.nw = len(d['wav'])
            self.ns = len(d['stokes'])
            self.dat = np.ma.filled(f.variables['profiles'][:], 0.0)
            self.wav = f.variables['wav'][:].data
            self.weights = np.ma.filled(f.variables['weights'][:], 1.0)
            self.pweights = np.ma.filled(f.variables['pixel_weights'][:], 1.0)
            self.psf = np.ma.filled(f.variables['psf'][:], 0.0) \
                if 'psf' in f.variables else np.zeros((0, 0))
            self.lts = np.ravel(f.variables['lts'][:]).astype('int32') \
                if 'lts' in f.variables else np.array([-1], 'int32')
            self.ltargs = np.ravel(np.ma.filled(f.variables['ltargs'][:], 0.)) \
                if 'ltargs' in f.variables else np.array([-1.0])
            self.ds = np.ma.filled(f.variables['ds'][:], 0.0) \
                if 'ds' in f.variables else np.zeros((0, 0, 0))

    def write(self, filename):
        from netCDF4 import Dataset
        with Dataset(filename, 'w') as f:
            f.createDimension('x', self.nx)
            f.createDimension('y', self.ny)
            f.createDimension('stokes', self.ns)
            f.createDimension('wav', self.nw)
            f.createDimension('time')
            f.createDimension('lts', max(1, self.lts.size))
            f.createDimension('ltargs', max(1, self.ltargs.size))
            f.createDimension('npy', self.psf.shape[0])
            f.createDimension('npx', self.psf.shape[1])
            f.createDimension('ndsn', self.ds.shape[0])
            f.createDimension('ndsy', self.ds.shape[1])
            f.createDimension('ndsx', self.ds.shape[2])

            prof = f.createVariable('profiles', 'f8',
                                    ('time', 'y', 'x', 'wav', 'stokes'))
            wav = f.createVariable('wav', 'f8', ('wav',))
            wei = f.createVariable('weights', 'f4', ('wav', 'stokes'))
            pwe = f.createVariable('pixel_weights', 'f4', ('time', 'y', 'x'))
            psf = f.createVariable('psf', 'f8', ('npy', 'npx'))
            lts = f.createVariable('lts', 'i4', ('lts',))
            lta = f.createVariable('ltargs', 'f8', ('ltargs',))
            ds = f.createVariable('ds', 'f8', ('ndsn', 'ndsy', 'ndsx'))

            for tt in range(self.nt):
                prof[tt] = self.dat[tt]
                pwe[tt] = self.pweights[tt]
            wav[:] = self.wav
            wei[:] = self.weights
            psf[:] = self.psf
            lts[:] = self.lts
            lta[:] = self.ltargs
            ds[:] = self.ds


# --------------------------------------------------------------------------- #
# input.cfg parsing
# --------------------------------------------------------------------------- #

def parse_regions(cfg_text):
    """Return list of dicts for the active region lines of an input.cfg."""
    regions = []
    for ln in cfg_text.splitlines():
        m = re.match(r'^\s*region\s*=\s*(.+)$', ln)
        if not m:
            continue
        parts = [p.strip() for p in m.group(1).split(',')]
        regions.append(dict(w0=float(parts[0]), dw=float(parts[1]),
                            nw=int(parts[2]), cscal=float(parts[3]),
                            inst=parts[4], inst_file=parts[5],
                            obs_file=parts[6]))
    return regions


def read_cfg_key(cfg_text, key):
    m = re.search(rf'^\s*{key}\s*=\s*(\S+)', cfg_text, re.M)
    return m.group(1) if m else None


# --------------------------------------------------------------------------- #
# tile geometry
# --------------------------------------------------------------------------- #

def _output_window_1d(m0, tn, d, s, rn, rd, rs):
    """Output-grid pixels [o0, o1) whose cells overlap the tile extent
    [m0*d + s - d/2, (m0+tn-1)*d + s + d/2)."""
    X0 = m0 * d + s - d / 2.0
    X1 = (m0 + tn - 1) * d + s + d / 2.0
    r = np.arange(rn)
    bot = r * rd + rs - rd / 2.0
    top = bot + rd
    keep = (top > X0) & (bot < X1)
    idx = np.where(keep)[0]
    if idx.size == 0:
        raise ValueError('tile does not overlap the output grid')
    return int(idx[0]), int(idx[-1]) + 1


def tile_obs(obs, mx0, my0, tnx, tny):
    """Cut one region's CoupledObs to the tile. Returns a new CoupledObs
    and the observed-pixel window (ox0, ox1, oy0, oy1)."""
    out = CoupledObs()
    out.wav = obs.wav.copy()
    out.weights = obs.weights.copy()
    out.psf = obs.psf.copy()
    out.nt, out.nw, out.ns = obs.nt, obs.nw, obs.ns

    if int(obs.lts[0]) < 0:
        # region observed on the fine/model grid
        ox0, ox1, oy0, oy1 = mx0, mx0 + tnx, my0, my0 + tny
        out.lts = obs.lts.copy()
        out.ltargs = obs.ltargs.copy()
        out.ds = np.zeros((0, 0, 0))
    else:
        if obs.lts.size != 1 or int(obs.lts[0]) != 10:
            raise NotImplementedError('only single LT-10 chains supported')
        a = obs.ltargs[:12].copy()
        rx, ry = int(a[0]), int(a[1])
        rdx, rdy, rsx, rsy = a[2], a[3], a[4], a[5]
        nx, ny = int(a[6]), int(a[7])
        dx, dy, sx, sy = a[8], a[9], a[10], a[11]

        ox0, ox1 = _output_window_1d(mx0, tnx, dx, sx, rx, rdx, rsx)
        oy0, oy1 = _output_window_1d(my0, tny, dy, sy, ry, rdy, rsy)

        new_args = np.array([
            ox1 - ox0, oy1 - oy0,                 # rx', ry'
            rdx, rdy,
            rsx + ox0 * rdx, rsy + oy0 * rdy,     # rsx', rsy'
            tnx, tny,                             # nx', ny'
            dx, dy,
            sx + mx0 * dx, sy + my0 * dy,         # sx', sy'
        ], dtype='float64')
        out.lts = np.array([10], 'int32')
        out.ltargs = new_args
        out.ds = obs.ds[:, my0:my0 + tny, mx0:mx0 + tnx].copy()

    out.dat = obs.dat[:, oy0:oy1, ox0:ox1].copy()
    out.pweights = obs.pweights[:, oy0:oy1, ox0:ox1].copy()
    out.ny, out.nx = out.dat.shape[1], out.dat.shape[2]

    # Coverage masking (extends the prepInv.py convention). Criteria are
    # computed with the FULL-grid operators, because the tile's own
    # operators are row-normalized and cannot see lost area:
    #  (a) model-space: the pixel's full-chain footprint (INCLUDING the
    #      spatial PSF, when present) lies in the tile
    #  (b) warp-space (coarse regions): its rebin support lies in the
    #      tile — warp-output cells beyond the tile edge can sample
    #      model pixels back INSIDE the tile (e.g. ds_x ~ -2), which the
    #      truncated tile warp cannot represent even though (a) holds.
    # Pixels failing either get weight 0; every pixel that participates
    # in the tile inversion then has an operator row identical to the
    # full-FOV one.
    has_psf = obs.psf.size and obs.psf.sum() >= 1e-12
    if int(out.lts[0]) == 10:
        covM, covW = _coverage_from_full(obs, mx0, my0, tnx, tny,
                                         ox0, ox1, oy0, oy1)
        out.pweights = out.pweights * \
            ((covM >= 0.9999) & (covW >= 0.9999))[None]
    elif has_psf:
        # fine-grid region with a PSF: rows whose footprint leaves the
        # tile renormalize differently and cannot be reproduced
        ind = np.zeros((obs.ny, obs.nx))
        ind[my0:my0 + tny, mx0:mx0 + tnx] = 1.0
        covF = _psf_coverage(obs.psf, ind)[oy0:oy1, ox0:ox1]
        out.pweights = out.pweights * (covF >= 0.9999)[None]
    return out, (ox0, ox1, oy0, oy1)


def _psf_coverage(psf, ind):
    """(P @ ind) for the row-normalized full-grid PSF operator, computed
    by correlation instead of a sparse matrix. ndimage's default kernel
    center (size//2) matches STiC's c = shape//2 convention (verified
    for odd and even sizes against degradation.psf_matrix)."""
    from scipy.ndimage import correlate
    p = np.asarray(psf, 'float64')
    p = p / p.sum()
    num = correlate(ind, p, mode='constant', cval=0.0)
    den = correlate(np.ones_like(ind), p, mode='constant', cval=0.0)
    return num / den


def _coverage_from_full(obs, mx0, my0, tnx, tny, ox0, ox1, oy0, oy1):
    """Per observed pixel of the tile window, the in-tile weight fraction
    of (a) the full-chain footprint (PSF + warp + rebin) in model space
    and (b) the rebin support in warp-output space, both from the
    FULL-grid operators."""
    a = obs.ltargs[:12]
    nx, ny = int(a[6]), int(a[7])
    W = dg.destretch_matrix(nx, ny, obs.ds[0], obs.ds[1])
    R = dg.rebin_matrix(int(a[0]), int(a[1]), a[2], a[3], a[4], a[5],
                        nx, ny, a[8], a[9], a[10], a[11])
    ind = np.zeros((ny, nx))
    ind[my0:my0 + tny, mx0:mx0 + tnx] = 1.0
    base = _psf_coverage(obs.psf, ind) \
        if (obs.psf.size and obs.psf.sum() >= 1e-12) else ind
    covM = (R @ (W @ base.ravel())).reshape(int(a[1]), int(a[0]))
    covW = (R @ ind.ravel()).reshape(int(a[1]), int(a[0]))
    return (covM[oy0:oy1, ox0:ox1], covW[oy0:oy1, ox0:ox1])


# --------------------------------------------------------------------------- #
# model slicing (coupled model files are the standard STiC model format)
# --------------------------------------------------------------------------- #

def tile_model(model_file, out_file, mx0, my0, tnx, tny):
    from . import stic_io
    m = stic_io.Model.read(model_file)
    out = stic_io.Model(nt=m.nt, ny=tny, nx=tnx, ndep=m.ndep)
    for v in stic_io.MODEL_VARS:
        a = stic_io._ATTR.get(v, v)
        getattr(out, a)[:] = getattr(m, a)[:, my0:my0 + tny, mx0:mx0 + tnx]
    out.tr_loc[:] = m.tr_loc[:, my0:my0 + tny, mx0:mx0 + tnx]
    out.tr_amp[:] = m.tr_amp[:, my0:my0 + tny, mx0:mx0 + tnx]
    out.tr_N[:] = m.tr_N[:, my0:my0 + tny, mx0:mx0 + tnx]
    out.write(out_file)


# --------------------------------------------------------------------------- #
# run-dir assembly
# --------------------------------------------------------------------------- #

AUX_LINKS = ['Atoms', 'Molecules', 'Atmos', 'atoms.input', 'kurucz.input',
             'keyword.input', 'molecules.input']


def extract_tile_rundir(src_dir, out_dir, mx0, my0, tnx, tny,
                        cfg_overrides=None, aux_search=(),
                        require_aux=True):
    """Build a self-contained tile run dir from a full coupled run dir.

    src_dir must contain input.cfg + the obs/inst/model files it names.
    aux_search: extra dirs to look for the AUX_LINKS files in.
    Returns a dict with tile metadata (windows per region).
    """
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(src_dir, 'input.cfg')) as f:
        cfg_text = f.read()
    regions = parse_regions(cfg_text)
    model_file = read_cfg_key(cfg_text, 'input_model')

    meta = dict(mx0=mx0, my0=my0, tnx=tnx, tny=tny, regions={})

    for reg in regions:
        obs = CoupledObs(os.path.join(src_dir, reg['obs_file']))
        tob, win = tile_obs(obs, mx0, my0, tnx, tny)
        tob.write(os.path.join(out_dir, reg['obs_file']))
        meta['regions'][reg['obs_file']] = dict(
            window=win, ny=tob.ny, nx=tob.nx,
            masked=float((tob.pweights == 0).mean()))
        # instrumental profile: spectral only, copy/link unchanged
        src_inst = os.path.join(src_dir, reg['inst_file'])
        dst_inst = os.path.join(out_dir, reg['inst_file'])
        if os.path.isfile(src_inst) and not os.path.exists(dst_inst):
            os.symlink(os.path.abspath(src_inst), dst_inst)

    tile_model(os.path.join(src_dir, model_file),
               os.path.join(out_dir, model_file), mx0, my0, tnx, tny)

    # auxiliary inputs — missing ones make RH segfault (unchecked fopen),
    # so fail loudly here instead
    missing = []
    for name in AUX_LINKS:
        dst = os.path.join(out_dir, name)
        if os.path.exists(dst):
            continue
        for d in (src_dir,) + tuple(aux_search):
            src = os.path.join(d, name)
            if os.path.exists(src):
                os.symlink(os.path.abspath(src), dst)
                break
        else:
            missing.append(name)
    if missing and require_aux:
        raise FileNotFoundError(
            f'auxiliary inputs not found in src_dir or aux_search: '
            f'{missing} — RH would segfault without them')

    # patched input.cfg
    overrides = dict(cfg_overrides or {})
    lines = []
    for ln in cfg_text.splitlines():
        m = re.match(r'^\s*([A-Za-z_0-9]+)\s*=', ln)
        if m and m.group(1) in overrides:
            key = m.group(1)
            lines.append(f'{key} = {overrides.pop(key)}')
        else:
            lines.append(ln)
    for k, v in overrides.items():
        lines.append(f'{k} = {v}')
    with open(os.path.join(out_dir, 'input.cfg'), 'w') as f:
        f.write('\n'.join(lines) + '\n')

    np.savez(os.path.join(out_dir, 'tile_meta.npz'), **{
        'mx0': mx0, 'my0': my0, 'tnx': tnx, 'tny': tny,
        'windows': np.array([meta['regions'][r['obs_file']]['window']
                             for r in regions]),
        'obs_files': np.array([r['obs_file'] for r in regions])})
    return meta


# --------------------------------------------------------------------------- #
# label harvesting
# --------------------------------------------------------------------------- #

def read_tile_labels(run_dir, atmos_name, apron):
    """Read an inverted tile atmosphere, discard the apron, and return
    (model, y_coords, x_coords) where the coordinates are absolute
    fine-grid indices of the kept columns.

    apron: number of fine-grid pixels stripped from every tile edge. It
    must cover PSF half-width + max destretch shift + one coarse rebin
    cell, so that both the operator-truncation region and the pixels
    starved of coarse-region constraints (weight-masked obs pixels) are
    excluded from the labels.
    """
    from . import stic_io
    meta = np.load(os.path.join(run_dir, 'tile_meta.npz'))
    m = stic_io.Model.read(os.path.join(run_dir, atmos_name))
    tny, tnx = int(meta['tny']), int(meta['tnx'])
    assert (m.ny, m.nx) == (tny, tnx), 'atmos does not match tile dims'
    a = int(apron)
    keep = np.s_[:, a:tny - a, a:tnx - a]
    out = stic_io.Model(nt=m.nt, ny=tny - 2 * a, nx=tnx - 2 * a,
                        ndep=m.ndep)
    for v in stic_io.MODEL_VARS:
        at = stic_io._ATTR.get(v, v)
        getattr(out, at)[:] = getattr(m, at)[keep]
    yy = int(meta['my0']) + a + np.arange(tny - 2 * a)
    xx = int(meta['mx0']) + a + np.arange(tnx - 2 * a)
    return out, yy, xx


def suggest_apron(run_dir_or_src):
    """Conservative apron from the stored operator specs: PSF half-width
    + max |destretch| + one coarse rebin cell, over all regions."""
    with open(os.path.join(run_dir_or_src, 'input.cfg')) as f:
        regions = parse_regions(f.read())
    apron = 0.0
    for reg in regions:
        o = CoupledObs(os.path.join(run_dir_or_src, reg['obs_file']))
        need = 0.0
        if o.psf.size:
            need += max(o.psf.shape) / 2.0
        if int(o.lts[0]) == 10:
            a = o.ltargs[:12]
            need += np.abs(o.ds).max() + max(a[2], a[3])
        apron = max(apron, need)
    return int(np.ceil(apron)) + 1
