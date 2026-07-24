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
    if want == 'cuda' and not torch.cuda.is_available():
        want = 'mps' if torch.backends.mps.is_available() else 'cpu'
    if want == 'mps' and not torch.backends.mps.is_available():
        want = 'cpu'
    return torch.device(want)


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
        model.eval()
        tot, n = 0.0, 0
        with torch.no_grad():
            for s in samples_v:
                nv = min(len(s['coords']), 8192)
                mu, lv = model(s, s['coords'][:nv], device)
                y = torch.as_tensor(s['Y'][:nv], device=device)
                tot += loss_fn(mu, lv, y, groups, 0.0).item() * nv
                n += nv
        model.train()
        return tot / max(n, 1)

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
            vl = val_loss() if samples_v else float(loss) * C.TILES_PER_STEP
            if vl < best:
                best, best_step = vl, it
                torch.save(model.state_dict(), ckpt)
            print(f'train[{k}] step {it + 1:6d}  val {vl:.4f}  best {best:.4f}')
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
    rng = np.random.default_rng(C.SEED)
    perm = rng.permutation(len(samples))
    nval = max(1, int(C.VAL_FRACTION * len(samples))) \
        if len(samples) > 3 else 0
    val = [samples[i] for i in perm[:nval]]
    trn = [samples[i] for i in perm[nval:]]
    print(f'train_ensemble: {len(trn)} train / {len(val)} val tiles')

    # shared metadata: written once; parallel --member jobs must not
    # clobber it (e.g. a different --members setting)
    meta_path = os.path.join(out_dir, 'meta.json')
    write_shared = member is None or not os.path.isfile(meta_path)
    if write_shared:
        np.savez(os.path.join(out_dir, 'encoding.npz'),
                 y_mean=meta['y_mean'], y_std=meta['y_std'],
                 target_names=np.asarray(meta['target_names']),
                 ltau_grid=C.LTAU_GRID,
                 **{f'stat_mu_{t}': meta['stats'][t][0]
                    for t in meta['stats']},
                 **{f'stat_sd_{t}': meta['stats'][t][1]
                    for t in meta['stats']})
        with open(meta_path, 'w') as f:
            json.dump(dict(enc_channels=meta['enc_channels'],
                           n_out=meta['n_out'],
                           c_feat=C.ENC_CHANNELS, enc_layers=C.ENC_LAYERS,
                           dec_hidden=list(C.DEC_HIDDEN), dropout=C.DROPOUT,
                           n_ensemble=n_ensemble or C.N_ENSEMBLE),
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
