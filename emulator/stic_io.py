"""
Minimal, self-contained I/O for STiC netCDF files.

Format-compatible with stic/pythontools sparsetools.py (profile and model
classes) but without the py2-era dependencies (matplotlib, imtools,
interp2d, ...). netCDF4 is imported lazily so numpy-only code paths
(sampling, dataset math) can run without it.

Conventions (same as sparsetools):
  profile: dims (time, y, x, wav, stokes); vars profiles, wav,
           weights(wav,stokes), pixel_weights(time,y,x).
           weights are noise sigmas; sigma >= WEIGHT_SENTINEL means
           "wavelength not observed, ignore".
  model:   dims (time, y, x, ndep); vars ltau500, z, temp, pgas, vlos,
           blong, bhor, azi, vturb, cmass [, nne, rho] and the
           transition_region_* scalars.
"""

import os
import numpy as np

WEIGHT_SENTINEL = 1.e10  # sigmas above this mean: point not used


def _nc():
    from netCDF4 import Dataset
    return Dataset


# --------------------------------------------------------------------------- #
# profiles
# --------------------------------------------------------------------------- #

class Profile:
    def __init__(self, nt=1, ny=1, nx=1, nw=1, ns=4, dtype='float64'):
        self.nt, self.ny, self.nx, self.nw, self.ns = nt, ny, nx, nw, ns
        self.dat = np.zeros((nt, ny, nx, nw, ns), dtype=dtype)
        self.wav = np.zeros(nw, dtype='float64')
        self.weights = np.ones((nw, ns), dtype='float64')
        self.pweights = np.ones((nt, ny, nx), dtype='float64')

    @classmethod
    def read(cls, filename, t0=0, t1=None):
        """Read a STiC profile file. Optionally only frames [t0:t1)."""
        Dataset = _nc()
        with Dataset(filename, 'r') as f:
            nt = len(f.dimensions['time'])
            ny = len(f.dimensions['y'])
            nx = len(f.dimensions['x'])
            nw = len(f.dimensions['wav'])
            ns = len(f.dimensions['stokes'])
            if t1 is None:
                t1 = nt
            p = cls(nt=t1 - t0, ny=ny, nx=nx, nw=nw, ns=ns)
            p.dat[:] = np.ma.filled(f.variables['profiles'][t0:t1], np.nan)
            p.wav[:] = f.variables['wav'][:]
            if 'weights' in f.variables:
                p.weights[:] = f.variables['weights'][:]
            if 'pixel_weights' in f.variables:
                p.pweights[:] = f.variables['pixel_weights'][t0:t1]
        return p

    def write(self, filename):
        Dataset = _nc()
        prec = 'f8' if self.dat.dtype == np.float64 else 'f4'
        with Dataset(filename, 'w', format='NETCDF4') as f:
            f.createDimension('x', self.nx)
            f.createDimension('y', self.ny)
            f.createDimension('stokes', self.ns)
            f.createDimension('wav', self.nw)
            f.createDimension('time')
            prof = f.createVariable('profiles', prec,
                                    ('time', 'y', 'x', 'wav', 'stokes'))
            wav = f.createVariable('wav', 'f8', ('wav',))
            wei = f.createVariable('weights', 'f4', ('wav', 'stokes'))
            pwe = f.createVariable('pixel_weights', 'f4', ('time', 'y', 'x'))
            for tt in range(self.nt):
                prof[tt] = self.dat[tt]
                pwe[tt] = self.pweights[tt]
            wav[:] = self.wav
            wei[:] = self.weights

    def used_mask(self):
        """(nw, ns) bool mask of wavelength/stokes points that carry data."""
        return self.weights < WEIGHT_SENTINEL


def read_profile_dims(filename):
    Dataset = _nc()
    with Dataset(filename, 'r') as f:
        return (len(f.dimensions['time']), len(f.dimensions['y']),
                len(f.dimensions['x']), len(f.dimensions['wav']),
                len(f.dimensions['stokes']))


def read_profile_pixels(filename, t_idx, y_idx, x_idx):
    """Gather individual pixels (t[i], y[i], x[i]) -> (npix, nw, ns).

    Reads frame by frame to keep memory bounded on big cubes.
    """
    Dataset = _nc()
    t_idx = np.asarray(t_idx); y_idx = np.asarray(y_idx); x_idx = np.asarray(x_idx)
    with Dataset(filename, 'r') as f:
        v = f.variables['profiles']
        nw = len(f.dimensions['wav']); ns = len(f.dimensions['stokes'])
        out = np.empty((t_idx.size, nw, ns), dtype='float64')
        for tt in np.unique(t_idx):
            sel = np.where(t_idx == tt)[0]
            frame = np.ma.filled(v[int(tt)], np.nan)  # (ny, nx, nw, ns)
            out[sel] = frame[y_idx[sel], x_idx[sel]]
    return out


# --------------------------------------------------------------------------- #
# models
# --------------------------------------------------------------------------- #

MODEL_VARS = ['ltau500', 'z', 'temp', 'pgas', 'rho', 'nne',
              'vlos', 'vturb', 'blong', 'bhor', 'azi', 'cmass']

# attribute name used on the Model object for each netCDF variable
_ATTR = {'ltau500': 'ltau', 'blong': 'Bln', 'bhor': 'Bho'}


class Model:
    def __init__(self, nt=1, ny=1, nx=1, ndep=1, dtype='float32'):
        self.nt, self.ny, self.nx, self.ndep = nt, ny, nx, ndep
        for v in MODEL_VARS:
            setattr(self, _ATTR.get(v, v),
                    np.zeros((nt, ny, nx, ndep), dtype=dtype))
        self.tr_loc = np.zeros((nt, ny, nx), dtype=dtype)
        self.tr_amp = np.ones((nt, ny, nx), dtype=dtype)
        self.tr_N = np.zeros((nt, ny, nx), dtype='int32')

    @classmethod
    def read(cls, filename, t0=0, t1=None):
        Dataset = _nc()
        with Dataset(filename, 'r') as f:
            nt = len(f.dimensions['time'])
            ny = len(f.dimensions['y'])
            nx = len(f.dimensions['x'])
            ndep = len(f.dimensions['ndep'])
            if t1 is None:
                t1 = nt
            m = cls(nt=t1 - t0, ny=ny, nx=nx, ndep=ndep)
            for v in MODEL_VARS:
                if v in f.variables:
                    getattr(m, _ATTR.get(v, v))[:] = \
                        np.ma.filled(f.variables[v][t0:t1], np.nan)
            for v, a in (('transition_region_loc', 'tr_loc'),
                         ('transition_region_scale', 'tr_amp'),
                         ('transition_region_nGrid', 'tr_N')):
                if v in f.variables:
                    getattr(m, a)[:] = f.variables[v][t0:t1]
        return m

    def write(self, filename, write_all=True):
        Dataset = _nc()
        with Dataset(filename, 'w', format='NETCDF4') as f:
            f.createDimension('y', self.ny)
            f.createDimension('x', self.nx)
            f.createDimension('ndep', self.ndep)
            f.createDimension('time')
            handles = {}
            varlist = list(MODEL_VARS)
            if not write_all:
                varlist = [v for v in varlist if v not in ('nne', 'rho')]
            for v in varlist:
                handles[v] = f.createVariable(v, 'f4',
                                              ('time', 'y', 'x', 'ndep'))
            h_loc = f.createVariable('transition_region_loc', 'f4',
                                     ('time', 'y', 'x'))
            h_amp = f.createVariable('transition_region_scale', 'f4',
                                     ('time', 'y', 'x'))
            h_N = f.createVariable('transition_region_nGrid', 'i4',
                                   ('time', 'y', 'x'))
            for tt in range(self.nt):
                for v in varlist:
                    handles[v][tt] = getattr(self, _ATTR.get(v, v))[tt]
                h_loc[tt] = self.tr_loc[tt]
                h_amp[tt] = self.tr_amp[tt]
                h_N[tt] = self.tr_N[tt]

    def extract_pixels(self, t_idx, y_idx, x_idx):
        """Gather pixels into a new Model with shape (1, 1, npix, ndep)."""
        n = len(t_idx)
        m = Model(nt=1, ny=1, nx=n, ndep=self.ndep)
        for v in MODEL_VARS:
            a = _ATTR.get(v, v)
            getattr(m, a)[0, 0] = getattr(self, a)[t_idx, y_idx, x_idx]
        m.tr_loc[0, 0] = self.tr_loc[t_idx, y_idx, x_idx]
        m.tr_amp[0, 0] = self.tr_amp[t_idx, y_idx, x_idx]
        m.tr_N[0, 0] = self.tr_N[t_idx, y_idx, x_idx]
        return m


# --------------------------------------------------------------------------- #
# chi2
# --------------------------------------------------------------------------- #

def chi2_map(obs, syn):
    """Reduced chi2 per pixel between two Profile objects (same grid).

    Uses the observation's noise weights; sentinel points are excluded.
    Returns (nt, ny, nx) array.
    """
    use = obs.used_mask()          # (nw, ns)
    sig = np.where(use, obs.weights, np.inf)
    r = (obs.dat - syn.dat) / sig  # broadcasting over (nt,ny,nx,nw,ns)
    r = np.where(np.isfinite(r), r, 0.0)
    ndata = max(int(use.sum()), 1)
    return (r * r).sum(axis=(3, 4)) / ndata
