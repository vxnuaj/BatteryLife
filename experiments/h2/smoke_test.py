"""H2 smoke test — CPU correctness check for physics-derived input channels.

Three checks, no DeepSpeed:
  1. ``add_physics_channels`` on synthetic [L,3,300]: shape -> [L,5,300], finite,
     and the reference cycle's deltaQ is ~0 (by construction).
  2. ``add_physics_channels`` on a REAL subset cell (MATR_b1c11.pkl): finite,
     sane magnitudes -- validates the feature on actual V/Q signals.
  3. CPMLP and CPTransformer with enc_in=5 (physics) AND enc_in=3 (base), for both
     intra encoders {linear, conv}: forward [B,100,C,300] -> [B,1], finite loss, backprop.
     Confirms H2 composes with H1 and that enc_in=3 still builds the SOTA models.

Run (from repo root):  python experiments/h2/smoke_test.py
"""
import os
import sys
import pickle
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)

from models import CPMLP, CPTransformer                 # noqa: E402
from utils.physics_features import add_physics_channels  # noqa: E402

B, S, L = 4, 100, 300


def cfg(enc_in, intra_encoder):
    return SimpleNamespace(
        d_model=128, d_ff=256, e_layers=4, d_layers=2,
        charge_discharge_length=L, early_cycle_threshold=S, dropout=0.0,
        n_heads=4, factor=3, activation='relu', output_num=1,
        enc_in=enc_in, intra_encoder=intra_encoder,
    )


def check_synthetic():
    curves = np.random.randn(S, 3, L).astype(np.float32)
    out = add_physics_channels(curves)
    assert out.shape == (S, 5, L), f"bad shape {out.shape}"
    assert np.isfinite(out).all(), "non-finite physics channels"
    # ref cycle (idx 0) deltaQ channel (last) should be ~0
    ref_dq = np.abs(out[0, 4, :]).max()
    assert ref_dq < 1e-3, f"reference-cycle deltaQ not ~0: {ref_dq}"
    print(f"[1] synthetic: [S,3,{L}] -> {out.shape}, finite, ref deltaQ max={ref_dq:.2e}  OK")


def check_real():
    p = os.path.join(_ROOT, "dataset/MATR/MATR_b1c11.pkl")
    if not os.path.exists(p):
        print("[2] real-data check SKIPPED (MATR sample not present)")
        return
    with open(p, 'rb') as f:
        d = pickle.load(f)
    nominal = d.get('nominal_capacity_in_Ah') or 1.1
    cd = d['cycle_data'][:S]
    curves = np.zeros((len(cd), 3, L), dtype=np.float32)
    for i, cyc in enumerate(cd):
        V = np.asarray(cyc['voltage_in_V'], float)
        Q = np.asarray(cyc['discharge_capacity_in_Ah'], float)
        I = np.asarray(cyc['current_in_A'], float)
        if len(V) < 2:
            continue
        xs = np.linspace(0, 1, len(V)); xt = np.linspace(0, 1, L)
        curves[i, 0] = np.interp(xt, xs, V) / (V.max() + 1e-9)
        curves[i, 1] = np.interp(xt, xs, I) / nominal
        curves[i, 2] = np.interp(xt, xs, Q) / nominal
    out = add_physics_channels(curves)
    assert out.shape == (len(cd), 5, L) and np.isfinite(out).all()
    dqdv, dq = out[:, 3, :], out[:, 4, :]
    print(f"[2] real MATR cell: {out.shape}, finite | "
          f"dQ/dV range=[{dqdv.min():.2f},{dqdv.max():.2f}] | deltaQ range=[{dq.min():.3f},{dq.max():.3f}]  OK")


def check_models():
    for Model, name in [(CPMLP, 'CPMLP'), (CPTransformer, 'CPTransformer')]:
        for enc_in in (3, 5):              # 3 = base (SOTA), 5 = physics
            for intra in ('linear', 'conv'):
                torch.manual_seed(0)
                model = Model.Model(cfg(enc_in, intra)).float()
                x = torch.randn(B, S, enc_in, L)
                mask = torch.ones(B, S)
                out = model(x, mask)
                assert out.shape == (B, 1), f"{name} enc_in={enc_in}/{intra}: {tuple(out.shape)}"
                loss = nn.functional.mse_loss(out, torch.randn(B, 1))
                assert torch.isfinite(loss)
                loss.backward()
                tag = 'base ' if enc_in == 3 else 'phys '
                print(f"[3] {name:14s} {tag}enc_in={enc_in} {intra:6s} -> out {tuple(out.shape)} "
                      f"loss={loss.item():.4f}  OK")


def main():
    print("=" * 78)
    print("H2 smoke test — physics input channels (CPU)")
    print("=" * 78)
    check_synthetic()
    check_real()
    check_models()
    print("ALL CHECKS PASSED")


if __name__ == '__main__':
    main()
