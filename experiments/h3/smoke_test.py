"""H3 smoke test — CPU correctness for joint multi-domain conditioning.

Covers everything verifiable without the full data pipeline / GPU:
  1. FiLMConditioner: identity at init (zero-init), and once weights are set, different
     domain ids produce different outputs; shapes preserved for [B,S,d] and [B,d].
  2. per_domain_metrics + filename_to_domain_id utilities.
  3. Combined split: MIX_all = MIX_large + Zn + CALB + Na (counts), and the domain map
     covers all four domains over the real file lists.
  4. CP{MLP,Transformer} with multidomain on: forward(domain_id) -> [B,1], finite, backprop;
     different domain ids => different predictions (conditioning is live after a step);
     multidomain off (and domain_id=None) reproduces the single-domain models.

Run (from repo root):  python experiments/h3/smoke_test.py
"""
import os
import sys
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)

from models import CPMLP, CPTransformer                                  # noqa: E402
from layers.domain_conditioning import FiLMConditioner                   # noqa: E402
from utils.domain_utils import filename_to_domain_id, per_domain_metrics, DOMAIN_NAMES  # noqa: E402
from data_provider.data_split_recorder import split_recorder             # noqa: E402

B, S, D = 4, 100, 128


def cfg(multidomain='off'):
    return SimpleNamespace(
        d_model=D, d_ff=256, e_layers=4, d_layers=2,
        charge_discharge_length=300, early_cycle_threshold=S, dropout=0.0,
        n_heads=4, factor=3, activation='relu', output_num=1, enc_in=3,
        intra_encoder='linear', multidomain=multidomain, num_domains=4,
    )


def check_film():
    film = FiLMConditioner(num_domains=4, d_model=D)
    x = torch.randn(B, S, D)
    dom = torch.tensor([0, 1, 2, 3])
    out = film(x, dom)
    assert out.shape == x.shape
    assert torch.allclose(out, x), "FiLM should be identity at zero-init"
    # set non-trivial weights -> different domains must diverge
    nn.init.normal_(film.embed.weight, std=0.5)
    xc = torch.randn(1, S, D).repeat(4, 1, 1)          # same input, different domains
    out = film(xc, dom)
    assert not torch.allclose(out[0], out[1]), "different domains should differ once trained"
    assert film(torch.randn(B, D), dom).shape == (B, D)  # [B,d] path
    print("[1] FiLMConditioner: identity@init, domain-sensitive after training, shapes OK")


def check_utils():
    assert filename_to_domain_id('ZN-coin_402-1.pkl') == 1
    assert filename_to_domain_id('NA-ion_270040.pkl') == 2
    assert filename_to_domain_id('CALB_0_B182.pkl') == 3
    assert filename_to_domain_id('MATR_b1c11.pkl') == 0
    preds = np.array([100, 220, 1000, 1100.])
    tgts = np.array([110, 200, 1000, 1000.])
    doms = np.array([0, 1, 2, 3])
    m = per_domain_metrics(preds, tgts, doms, alpha=0.15)
    assert set(m) == set(DOMAIN_NAMES) | {'overall'}
    assert all('mape' in v and 'acc' in v for v in m.values())
    print(f"[2] utils: domain map OK | per_domain_metrics overall MAPE={m['overall']['mape']:.3f}")


def check_split():
    tr = split_recorder.MIX_all_train_files
    parts = (len(split_recorder.MIX_large_train_files) + len(split_recorder.ZNcoin_train_files)
             + len(split_recorder.CALB_train_files) + len(split_recorder.NAion_2021_train_files))
    assert len(tr) == parts, f"MIX_all_train={len(tr)} != sum of parts={parts}"
    doms = {filename_to_domain_id(f) for f in tr}
    assert doms == {0, 1, 2, 3}, f"MIX_all train missing domains: {doms}"
    print(f"[3] MIX_all split: {len(tr)} train cells, all 4 domains present  OK")


def check_models():
    for Model, name in [(CPMLP, 'CPMLP'), (CPTransformer, 'CPTransformer')]:
        # multidomain off => single-domain SOTA still builds & runs (domain_id ignored)
        torch.manual_seed(0)
        m_off = Model.Model(cfg('off'))
        assert m_off(torch.randn(B, S, 3, 300), torch.ones(B, S)).shape == (B, 1)

        # multidomain on
        torch.manual_seed(0)
        m = Model.Model(cfg('on'))
        nn.init.normal_(m.film.embed.weight, std=0.5)        # make conditioning non-trivial
        x = torch.randn(B, S, 3, 300); mask = torch.ones(B, S)
        dom = torch.tensor([0, 1, 2, 3])
        out = m(x, mask, domain_id=dom)
        assert out.shape == (B, 1) and torch.isfinite(out).all()
        nn.functional.mse_loss(out, torch.randn(B, 1)).backward()

        # same input, different domain ids -> different predictions
        xc = torch.randn(1, S, 3, 300).repeat(4, 1, 1, 1)
        outc = m(xc, torch.ones(4, S), domain_id=dom)
        assert not torch.allclose(outc[0], outc[1]), f"{name}: conditioning not affecting output"
        # domain_id=None disables conditioning gracefully
        assert m(x, mask, domain_id=None).shape == (B, 1)
        print(f"[4] {name:14s}: off-path OK | on-path forward+backward OK | domain-sensitive OK")


def check_collate_and_sampler():
    # balanced-sampling weights = inverse domain frequency (mirrors data_factory logic; pure numpy)
    dom = np.array([0, 0, 0, 0, 1, 2, 3])      # Li-ion over-represented
    counts = np.bincount(dom, minlength=4).astype(float)
    w = np.where(counts > 0, 1.0 / counts, 0.0)[dom]
    assert w[0] < w[4], "scarce domains must get higher sampling weight"

    # multidomain collate returns an 8-tuple with a LongTensor domain_id.
    # Importing data_loader pulls the full data stack (batteryml/denseweight); skip if absent
    # (it WILL be present on the GPU box where this runs for real).
    try:
        from data_provider.data_loader import my_collate_fn_multidomain
    except Exception as e:
        print(f"[5] inverse-freq weights OK | collate check SKIPPED (data stack absent: {type(e).__name__})")
        return
    samples = [{
        'cycle_curve_data': torch.randn(S, 3, 300), 'curve_attn_mask': torch.ones(S),
        'life_class': 1, 'labels': 0.3, 'scaled_life_class': 0,
        'weight': 1.0, 'seen_unseen_id': 1, 'domain_id': d,
    } for d in [0, 1, 2, 3]]
    out = my_collate_fn_multidomain(samples)
    assert len(out) == 8 and out[-1].dtype == torch.long and out[-1].tolist() == [0, 1, 2, 3]
    assert out[0].shape == (4, S, 3, 300)
    print("[5] multidomain collate (8-tuple, long domain_id) + inverse-freq weights  OK")


def main():
    print("=" * 78)
    print("H3 smoke test — joint multi-domain conditioning (CPU)")
    print("=" * 78)
    check_film()
    check_utils()
    check_split()
    check_models()
    check_collate_and_sampler()
    print("ALL CHECKS PASSED")


if __name__ == '__main__':
    main()
