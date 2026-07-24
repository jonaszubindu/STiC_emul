"""
Python/torch reimplementation of coupled-STiC's spatial degradation
operators (src/linear_transformations.hpp + coupled_inversion.cc).

Each spectral region r of the coupled inversion carries a sparse operator

    D_r = LT_n ... LT_1 . PSF        (fltd in the C++ code)

mapping a monochromatic image on the fine model grid (ny_model, nx_model)
to that region's observed grid. The operator specification is stored by
pycSTiC inside each obs_<region>.nc file (variables psf, lts, ltargs, ds);
this module reads that specification and rebuilds the identical matrix
with scipy, exposing dense/torch application.

Conventions (identical to the C++):
- images are flattened y-major: index = iy * nx + ix
- grid cells: pixel k of a 1-D grid (n, d, s) covers [k*d + s - d/2,
  k*d + s + d/2]
- all operators are row-normalized over their in-bounds entries
"""

import numpy as np
import scipy.sparse as sp


# --------------------------------------------------------------------------- #
# 1-D building blocks (the 2-D operators are separable)
# --------------------------------------------------------------------------- #

def _overlap_1d(rn, rd, rs, n, d, s):
    """Area-overlap weights between output cells (rn, rd, rs) and input
    cells (n, d, s). Returns a (rn, n) CSR matrix, NOT normalized.

    Mirrors get_pnt + get_rebin_area in linear_transformations.hpp:
    cell k covers [k*d + s - d/2, k*d + s + d/2]; entries kept if the
    overlap length is > 0 in this dimension.
    """
    rbot = np.arange(rn) * rd + rs - rd / 2.0
    rtop = rbot + rd
    bot = np.arange(n) * d + s - d / 2.0
    top = bot + d
    lo = np.maximum(bot[None, :], rbot[:, None])
    hi = np.minimum(top[None, :], rtop[:, None])
    w = hi - lo
    w[w <= 0] = 0.0
    return sp.csr_matrix(w)


def _row_normalize(m):
    m = m.tocsr()
    norm = np.asarray(m.sum(axis=1)).ravel()
    inv = np.zeros_like(norm)
    nz = norm != 0
    inv[nz] = 1.0 / norm[nz]
    return sp.diags(inv) @ m


# --------------------------------------------------------------------------- #
# operators
# --------------------------------------------------------------------------- #

def psf_matrix(nx, ny, psf):
    """PSF as a sparse (nx*ny, nx*ny) operator on the model grid.

    Mirrors lnrtr::get_psf_as_lt with psflim=0, normalize=true: output
    pixel (iy, ix) collects input pixels (iy - cy + jy, ix - cx + jx)
    with weight psf[jy, jx] (correlation-style placement, cx = npx//2),
    row-normalized over the in-bounds part of the footprint.
    """
    psf = np.asarray(psf, dtype='float64')
    psf = psf / psf.sum()          # same pre-normalization as the C++ caller
    npy, npx = psf.shape
    cy, cx = npy // 2, npx // 2

    oy = np.arange(npy) - cy       # footprint offsets
    ox = np.arange(npx) - cx

    iy, ix = np.divmod(np.arange(ny * nx), nx)
    jy, jx = np.broadcast_arrays(iy[:, None, None] + oy[None, :, None],
                                 ix[:, None, None] + ox[None, None, :])
    ok = (jy >= 0) & (jy < ny) & (jx >= 0) & (jx < nx)   # (npix, npy, npx)

    rows = np.broadcast_to(np.arange(ny * nx)[:, None, None], ok.shape)[ok]
    cols = (jy * nx + jx)[ok]
    vals = np.broadcast_to(psf[None, :, :], ok.shape)[ok]

    m = sp.csr_matrix((vals, (rows, cols)), shape=(ny * nx, ny * nx))
    return _row_normalize(m)


def rebin_matrix(rx, ry, rdx, rdy, rsx, rsy, nx, ny, dx, dy, sx, sy):
    """Area-overlap rebin from grid (nx, ny, dx, dy, sx, sy) to grid
    (rx, ry, rdx, rdy, rsx, rsy) as a sparse (rx*ry, nx*ny) matrix.

    Mirrors lnrtr::sparse_prebin. The geometry is separable, and the
    per-output-pixel normalization factorizes ((sum_y w_y)(sum_x w_x)),
    so the 2-D operator is exactly kron(Wy_norm, Wx_norm).
    """
    wx = _row_normalize(_overlap_1d(rx, rdx, rsx, nx, dx, sx))
    wy = _row_normalize(_overlap_1d(ry, rdy, rsy, ny, dy, sy))
    return sp.kron(wy, wx, format='csr')


def destretch_matrix(nx, ny, dsx, dsy):
    """Backward-warp operator on an (ny, nx) grid: output pixel p reads
    the input at p + (dsy[p], dsx[p]) with bilinear (area) weights.

    Mirrors lnrtr::sparse_pdestretch (unit cells, per-output-pixel shift
    of the input grid origin). Out-of-bounds neighbors are dropped and
    each row is renormalized, exactly like the C++.
    """
    dsx = np.asarray(dsx, dtype='float64').ravel()
    dsy = np.asarray(dsy, dtype='float64').ravel()
    npix = nx * ny
    ry, rx = np.divmod(np.arange(npix), nx)

    gx = rx + dsx                  # sampling position in input index space
    gy = ry + dsy
    ix0 = np.floor(gx)
    iy0 = np.floor(gy)
    fx = gx - ix0
    fy = gy - iy0

    rows, cols, vals = [], [], []
    for ox, wxs in ((0, 1.0 - fx), (1, fx)):
        for oy, wys in ((0, 1.0 - fy), (1, fy)):
            jx = (ix0 + ox).astype(np.int64)
            jy = (iy0 + oy).astype(np.int64)
            w = wxs * wys
            ok = (jx >= 0) & (jx < nx) & (jy >= 0) & (jy < ny) & (w > 0)
            rows.append(np.arange(npix)[ok])
            cols.append((jy * nx + jx)[ok])
            vals.append(w[ok])

    m = sp.csr_matrix((np.concatenate(vals),
                       (np.concatenate(rows), np.concatenate(cols))),
                      shape=(npix, npix))
    return _row_normalize(m)


def bintrim_matrix(nix, niy, dx, dy, cx, cy, nxe, nye):
    """LT type 4 (Rebin+Trim), mirrors build_lt_sparsematrix ltid==4.

    Output cell (sx, sy): [sx*dx + cx, sx*dx + cx + dx] (NOT center-
    shifted); input cells are unit cells [i, i+1]. Output dims are
    (round(nxe) - 1, round(nye) - 1) — the args store the number of cell
    edges. Separable, row-normalized.
    """
    nox = int(round(nxe)) - 1
    noy = int(round(nye)) - 1
    # output cell [s*d + c, s*d + c + d] == centered convention with
    # origin shifted by +d/2:
    wx = _row_normalize(_overlap_1d(nox, dx, cx + dx / 2.0, nix, 1.0, 0.5))
    wy = _row_normalize(_overlap_1d(noy, dy, cy + dy / 2.0, niy, 1.0, 0.5))
    return sp.kron(wy, wx, format='csr')


# --------------------------------------------------------------------------- #
# region operator from an obs_<region>.nc file
# --------------------------------------------------------------------------- #

LT_NARGS = {1: 2, 2: 4, 3: 3, 4: 6, 10: 12}


def _read_obs_spec(filename):
    from netCDF4 import Dataset
    with Dataset(filename, 'r') as f:
        out = {
            'ny': len(f.dimensions['y']),
            'nx': len(f.dimensions['x']),
            'psf': None, 'lts': None, 'ltargs': None, 'ds': None,
        }
        for v in ('psf', 'lts', 'ltargs', 'ds'):
            if v in f.variables and 0 not in f.variables[v].shape:
                out[v] = np.ma.filled(f.variables[v][:], 0.0)
    return out


def build_region_operator(obs_file, model_nx, model_ny):
    """Rebuild fltd for one region: D = LT_n ... LT_1 . PSF.

    obs_file: the region's observation netCDF (pycSTiC format).
    model_nx/model_ny: fine model-grid dimensions (iput.nx/ny in STiC).
    Returns a scipy CSR matrix of shape (ny_obs*nx_obs, model_ny*model_nx).
    """
    spec = _read_obs_spec(obs_file)

    # PSF (identity if absent, like the C++ default 1x1 PSF of 1.0)
    if spec['psf'] is not None and np.sum(spec['psf']) >= 1e-12:
        D = psf_matrix(model_nx, model_ny, spec['psf'])
    else:
        D = sp.identity(model_nx * model_ny, format='csr')

    lts = spec['lts']
    if lts is None or int(np.ravel(lts)[0]) < 0:
        return D.tocsr()

    ltargs = np.ravel(spec['ltargs']).astype('float64')
    off = 0
    nix, niy = model_nx, model_ny   # current input dims (C++ tracks these)
    for ltid in np.ravel(lts).astype(int):
        nargs = LT_NARGS[ltid]
        a = ltargs[off:off + nargs]
        off += nargs
        if ltid == 10:
            # a = [rx, ry, rdx, rdy, rsx, rsy, nx, ny, dx, dy, sx, sy]
            nx, ny = int(a[6]), int(a[7])
            ds = np.asarray(spec['ds'], dtype='float64').reshape(2, ny, nx)
            W = destretch_matrix(nx, ny, ds[0], ds[1])
            R = rebin_matrix(int(a[0]), int(a[1]), a[2], a[3], a[4], a[5],
                             nx, ny, a[8], a[9], a[10], a[11])
            it = R @ W
            nix, niy = int(a[0]), int(a[1])
        elif ltid == 4:
            # a = [dx, dy, cx, cy, n_x_edges, n_y_edges]
            it = bintrim_matrix(nix, niy, a[0], a[1], a[2], a[3], a[4], a[5])
            nix, niy = int(round(a[4])) - 1, int(round(a[5])) - 1
        else:
            raise NotImplementedError(
                f'LT type {ltid} is not implemented in STiC itself '
                '(build_lt_sparsematrix supports only 4 and 10)')
        D = (it @ D).tocsr()
    return D


# --------------------------------------------------------------------------- #
# torch wrapper
# --------------------------------------------------------------------------- #

class TorchRegionOperator:
    """D_r as a torch sparse CSR tensor with batched application.

    apply(x): x of shape (..., model_ny, model_nx) -> (..., ny_obs, nx_obs).
    Differentiable w.r.t. x (sparse mm), so it can sit inside a loss.
    """

    def __init__(self, D_scipy, obs_shape, model_shape,
                 device='cpu', dtype=None):
        import torch
        self.torch = torch
        self.obs_shape = tuple(obs_shape)      # (ny_obs, nx_obs)
        self.model_shape = tuple(model_shape)  # (ny_model, nx_model)
        dtype = dtype or torch.float32
        D = D_scipy.tocsr()
        self.D = torch.sparse_csr_tensor(
            torch.from_numpy(D.indptr.astype(np.int64)),
            torch.from_numpy(D.indices.astype(np.int64)),
            torch.from_numpy(D.data).to(dtype),
            size=D.shape, device=device, dtype=dtype)

    @classmethod
    def from_obs_file(cls, obs_file, model_nx, model_ny, **kw):
        from netCDF4 import Dataset
        with Dataset(obs_file) as f:
            obs_shape = (len(f.dimensions['y']), len(f.dimensions['x']))
        D = build_region_operator(obs_file, model_nx, model_ny)
        return cls(D, obs_shape, (model_ny, model_nx), **kw)

    def apply(self, x):
        torch = self.torch
        lead = x.shape[:-2]
        v = x.reshape(-1, self.model_shape[0] * self.model_shape[1])
        out = torch.sparse.mm(self.D, v.T.to(self.D.dtype)).T
        return out.reshape(*lead, *self.obs_shape)
