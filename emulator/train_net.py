"""
Train the coupled-emulator ensemble on harvested tile records.

Usage:
  python train_net.py <label_dir> [<label_dir> ...] --out <ckpt_dir>
"""

import os
import json
import time
import argparse
import numpy as np
import torch

from . import emu_config as C
from . import nn_dataset as nd
from .net import CoupledEmulator, loss_fn, make_depth_groups


def _device():
    want = C.DEVICE
    if want == 'auto':
        want = 'cuda' if torch.cuda.is_available() else (
            'mps' if torch.backends.mps.is_available() else 'cpu')
    if want == 'cuda' and not torch.cuda.is_available():
        print('WARNING: cuda requested but unavailable; using cpu')
        want = 'cpu'
    if want == 'mps' and not torch.backends.mps.is_available():
        want = 'cpu'
    print(f'torch {torch.__version__} (cuda {torch.version.cuda}) '
          f'from {torch.__file__}')
    if want == 'cuda':
        # torch only WARNS when it lacks kernels for the GPU and then
        # crashes at the first kernel; run one to fail early and clearly
        try:
            (torch.ones(8, device='cuda') * 2).sum().item()
        except RuntimeError as e:
            raise RuntimeError(
                f'this PyTorch build cannot run on '
                f'{torch.cuda.get_device_name()} (capability '
                f'{torch.cuda.get_device_capability()}; build supports '
                f'{torch.cuda.get_arch_list()}). Install a PyTorch build '
                f'for this GPU into the active env, and set '
                f'PYTHONNOUSERSITE=1 if ~/.local shadows it.') from e
    return torch.device(want)


def _subset(s, mask):
    """View of a sample restricted to some of its label pixels (the
    region images are shared, not copied)."""
    d = {k: v for k, v in s.items() if k not in ('coords', 'Y', 'split')}
    d['coords'] = s['coords'][mask]
    d['Y'] = s['Y'][mask]
    return d


def train_member(k, samples_t, samples_v, meta, out_dir, device,
                 steps=None):
    torch.manual_seed(C.SEED + 1000 * k)
    rng = np.random.default_rng(C.SEED + 1000 * k)
    steps = steps or C.STEPS
    groups = make_depth_groups(meta['target_names'])

    model = CoupledEmulator(meta['enc_channels'], meta['n_out'],
                            c_feat=C.ENC_CHANNELS, enc_layers=C.ENC_LAYERS,
                            dec_hidden=C.DEC_HIDDEN,
                            dropout=C.DROPOUT).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=C.LR,
                            weight_decay=C.WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)

    def val_loss():
        """(nll, mse) on the held-out pixels; mse is the error of the
        predicted values in standardized target units."""
        model.eval()
        nll, mse, n = 0.0, 0.0, 0
        with torch.no_grad():
            for s in samples_v:
                nv = min(len(s['coords']), 8192)
                mu, lv = model(s, s['coords'][:nv], device)
                y = torch.as_tensor(s['Y'][:nv], device=device)
                nll += loss_fn(mu, lv, y, groups, 0.0).item() * nv
                mse += ((mu - y) ** 2).mean().item() * nv
                n += nv
        model.train()
        return nll / max(n, 1), mse / max(n, 1)

    best, best_step = np.inf, 0
    ckpt = os.path.join(out_dir, f'member_{k}.pt')
    model.train()
    for it in range(steps):
        opt.zero_grad()
        for _ in range(C.TILES_PER_STEP):
            s = samples_t[rng.integers(len(samples_t))]
            n = len(s['coords'])
            if n > C.MAX_PIX_PER_TILE:
                idx = rng.choice(n, C.MAX_PIX_PER_TILE, replace=False)
            else:
                idx = np.arange(n)
            mu, lv = model(s, s['coords'][idx], device)
            y = torch.as_tensor(s['Y'][idx], device=device)
            loss = loss_fn(mu, lv, y, groups, C.DEPTH_SMOOTH_LAMBDA) \
                / C.TILES_PER_STEP
            loss.backward()
        opt.step()
        sched.step()
        if (it + 1) % 200 == 0 or it == steps - 1:
            if samples_v:
                v_nll, v_mse = val_loss()
                vl = v_mse if C.EARLY_STOP_ON == 'mse' else v_nll
                txt = f'held-out nll {v_nll:8.4f}  mse {v_mse:.4f}'
            else:
                vl = loss.item() * C.TILES_PER_STEP
                txt = f'train loss {vl:.4f}'
            if vl < best:
                best, best_step = vl, it
                torch.save(model.state_dict(), ckpt)
            print(f'train[{k}] step {it + 1:6d}  {txt}  '
                  f'best {C.EARLY_STOP_ON if samples_v else "loss"} '
                  f'{best:.4f} @ {best_step + 1}')
            if it - best_step > C.PATIENCE_STEPS:
                print(f'train[{k}]: early stop')
                break
    if not os.path.isfile(ckpt):
        torch.save(model.state_dict(), ckpt)
    return best


def train_ensemble(label_dirs, out_dir, steps=None, n_ensemble=None,
                   member=None):
    os.makedirs(out_dir, exist_ok=True)
    device = _device()
    print('device:', device)
    samples, meta = nd.build_dataset(label_dirs, chi2_max=C.CHI2_MAX)
    trn, val = [], []
    for s in samples:
        for code, lst in ((0, trn), (1, val)):
            m = s['split'] == code
            if m.any():
                lst.append(_subset(s, m))
    n_tr = sum(len(s['coords']) for s in trn)
    n_va = sum(len(s['coords']) for s in val)
    n_gu = sum(int((s['split'] == 2).sum()) for s in samples)
    print(f'train_ensemble: label pixels  train {n_tr}  held-out {n_va}  '
          f'guard (unused) {n_gu}  from {len(samples)} record(s)')
    if not trn:
        raise ValueError('no training pixels left after the spatial split')
    if not val:
        print('train_ensemble: WARNING no held-out pixels; early stopping '
              'falls back to the training loss')

    # shared metadata: written once; parallel --member jobs must not
    # clobber it (e.g. a different --members setting)
    meta_path = os.path.join(out_dir, 'meta.json')
    write_shared = member is None or not os.path.isfile(meta_path)
    if not write_shared:
        # another member already wrote the shared files: they must describe
        # THIS training setup, or prediction would decode with the wrong
        # normalization (e.g. retraining into an old checkpoint dir)
        with open(meta_path) as f:
            old = json.load(f)
        mine = dict(n_out=meta['n_out'],
                    chi2_threshold=meta['chi2_threshold'],
                    split=meta['split'], n_train_px=n_tr,
                    target_repr=meta['target_repr'],
                    pol_over_i=meta['pol_over_i'],
                    b_perp_vector=meta['b_perp_vector'])
        diff = [k for k, v in mine.items() if old.get(k) != v]
        if diff:
            raise ValueError(
                f'{out_dir} already holds an ensemble trained with a '
                f'different setup ({", ".join(diff)} differ). Use a new '
                f'--out directory.')
    if write_shared:
        np.savez(os.path.join(out_dir, 'encoding.npz'),
                 y_mean=meta['y_mean'], y_std=meta['y_std'],
                 target_names=np.asarray(meta['target_names']),
                 target_ltau=np.asarray(meta['target_ltau']),
                 target_extrap=np.asarray(meta['target_extrap']),
                 depth_grid=np.asarray(meta['depth_grid']),
                 pol_over_i=meta['pol_over_i'],
                 ltau_grid=C.LTAU_GRID,
                 **{f'chan_wav_{t}': w
                    for t, (w, s) in meta['channels'].items()},
                 **{f'chan_stokes_{t}': s
                    for t, (w, s) in meta['channels'].items()},
                 **{f'stat_mu_{t}': meta['stats'][t][0]
                    for t in meta['stats']},
                 **{f'stat_sd_{t}': meta['stats'][t][1]
                    for t in meta['stats']})
        with open(meta_path, 'w') as f:
            json.dump(dict(enc_channels=meta['enc_channels'],
                           n_out=meta['n_out'],
                           c_feat=C.ENC_CHANNELS, enc_layers=C.ENC_LAYERS,
                           dec_hidden=list(C.DEC_HIDDEN), dropout=C.DROPOUT,
                           n_ensemble=n_ensemble or C.N_ENSEMBLE,
                           chi2_threshold=meta['chi2_threshold'],
                           chi2_how=meta['chi2_how'],
                           split=meta['split'],
                           target_repr=meta['target_repr'],
                           nodes=meta['nodes'],
                           pol_over_i=meta['pol_over_i'],
                           b_perp_vector=meta['b_perp_vector'],
                           early_stop_on=C.EARLY_STOP_ON,
                           n_train_px=n_tr, n_val_px=n_va, n_guard_px=n_gu),
                      f, indent=1)

    t0 = time.time()
    ks = [member] if member is not None \
        else list(range(n_ensemble or C.N_ENSEMBLE))
    for k in ks:
        best = train_member(k, trn, val, meta, out_dir, device, steps=steps)
        print(f'member {k}: best val {best:.4f}')
    print(f'train_ensemble: done in {(time.time() - t0) / 60:.1f} min '
          f'-> {out_dir}')
    return out_dir


def load_ensemble(ckpt_dir, device):
    with open(os.path.join(ckpt_dir, 'meta.json')) as f:
        meta = json.load(f)
    models = []
    for k in range(meta['n_ensemble']):
        m = CoupledEmulator(meta['enc_channels'], meta['n_out'],
                            c_feat=meta['c_feat'],
                            enc_layers=meta['enc_layers'],
                            dec_hidden=meta['dec_hidden'],
                            dropout=meta['dropout']).to(device)
        p = os.path.join(ckpt_dir, f'member_{k}.pt')
        m.load_state_dict(torch.load(p, map_location=device))
        m.eval()
        models.append(m)
    enc = np.load(os.path.join(ckpt_dir, 'encoding.npz'))
    return models, enc


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('label_dirs', nargs='+')
    ap.add_argument('--out', required=True)
    ap.add_argument('--steps', type=int, default=None)
    ap.add_argument('--members', type=int, default=None,
                    help='total ensemble size (default emu_config)')
    ap.add_argument('--member', type=int, default=None,
                    help='train only this member index (one job per GPU: '
                         'CUDA_VISIBLE_DEVICES=k ... --member k)')
    a = ap.parse_args()
    train_ensemble(a.label_dirs, a.out, steps=a.steps, n_ensemble=a.members,
                   member=a.member)
