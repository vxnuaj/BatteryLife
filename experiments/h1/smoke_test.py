"""H1 smoke test — CPU correctness check for the conv intra-cycle encoder.

Bypasses DeepSpeed/accelerate entirely. Builds CPMLP and CPTransformer with both
``--intra_encoder linear`` (current SOTA) and ``conv`` (Hypothesis H1), then on
synthetic data matching the real loader contract ``[B, 100, 3, 300]`` checks that:
  1. each variant produces a [B, 1] prediction,
  2. the loss is finite and backprop works,
  3. the conv intra-encoder's parameter count stays under the linear one's
     (115,328 = Linear(900->128)), so any later win is structural, not capacity.

Run (from repo root):  python experiments/h1/smoke_test.py
"""
import os
import sys
from types import SimpleNamespace

import torch
import torch.nn as nn

# repo root is three levels up: experiments/h1/smoke_test.py -> experiments/h1 -> experiments -> root
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)

from models import CPMLP, CPTransformer            # noqa: E402
from layers.intra_encoders import ConvIntraEncoder  # noqa: E402

B, S, C, L = 4, 100, 3, 300   # batch, cycles, vars(V/I/Q), resampled points


def base_cfg(intra_encoder):
    return SimpleNamespace(
        d_model=128, d_ff=256, e_layers=4, d_layers=2,
        charge_discharge_length=L, early_cycle_threshold=S, dropout=0.0,
        n_heads=4, factor=3, activation='relu', output_num=1,
        intra_encoder=intra_encoder,
    )


def n_params(module):
    return sum(p.numel() for p in module.parameters())


def make_batch(pad=True):
    x = torch.randn(B, S, C, L)
    mask = torch.ones(B, S)
    if pad:                       # zero out the tail cycles of one sample => exercise masking
        mask[0, 60:] = 0
        x[0, 60:] = 0
    y = torch.randn(B, 1)
    return x, mask, y


def run_variant(Model, name, intra_encoder):
    torch.manual_seed(0)
    model = Model.Model(base_cfg(intra_encoder)).float()
    x, mask, y = make_batch()
    out = model(x, mask)
    assert out.shape == (B, 1), f"{name}/{intra_encoder}: bad output shape {tuple(out.shape)}"
    loss = nn.functional.mse_loss(out, y)
    assert torch.isfinite(loss), f"{name}/{intra_encoder}: non-finite loss"
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert all(g is not None and torch.isfinite(g).all() for g in grads), \
        f"{name}/{intra_encoder}: bad gradients"
    total = n_params(model)
    intra = n_params(model.conv_intra) if intra_encoder == 'conv' \
        else n_params(model.intra_embed)
    print(f"  {name:14s} {intra_encoder:6s} | out={tuple(out.shape)} "
          f"loss={loss.item():.4f} | total params={total:,} | intra params={intra:,}")
    return intra


def main():
    print("=" * 78)
    print("H1 smoke test — conv vs linear intra-cycle encoder (CPU)")
    print("=" * 78)

    # 1) ConvIntraEncoder in isolation
    enc = ConvIntraEncoder(num_var=C, fixed_len=L, d_model=128)
    z = enc(torch.randn(2, S, C, L))
    assert z.shape == (2, S, 128), f"ConvIntraEncoder bad shape {tuple(z.shape)}"
    z.sum().backward()
    print(f"ConvIntraEncoder: in [2,{S},{C},{L}] -> out {tuple(z.shape)} | "
          f"params={n_params(enc):,}  (budget ceiling = 115,328)")
    assert n_params(enc) <= 115_328, "conv encoder exceeds linear param budget!"
    print()

    # 2) full models, both variants
    intra_linear = intra_conv = None
    for Model, name in [(CPMLP, 'CPMLP'), (CPTransformer, 'CPTransformer')]:
        intra_linear = run_variant(Model, name, 'linear')
        intra_conv = run_variant(Model, name, 'conv')

    print()
    print(f"budget check: conv intra params {intra_conv:,} <= linear intra params "
          f"{intra_linear:,}  -> {'OK' if intra_conv <= intra_linear else 'FAIL'}")
    print("ALL CHECKS PASSED")


if __name__ == '__main__':
    main()
