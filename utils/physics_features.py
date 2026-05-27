"""Physics-derived per-cycle channels for Hypothesis H2.

Augments each battery's cycling curves with two domain-derived channels on top of
the raw [V, I, Q]:

  * ``dqdv``  : incremental capacity dQ/dV (per cycle), the classic IC-curve signal
                whose peaks track electrode phase transitions / degradation.
  * ``deltaq``: capacity-voltage difference vs. a reference early cycle, ΔQ(V).
                Severson et al. (2019) predict log cycle-life from Var(ΔQ(V)) with a
                *linear* model at ~9% error -- the single most validated BLP feature.

Contract (matches data_loader.py:618-626):
    input  ``curves`` : np.ndarray [L, 3, P], channel order [V, I, Q],
            P = charge_len + discharge_len with first half = charge, second = discharge.
    output            : np.ndarray [L, 3 + n_new, P]

Padded (all-zero) cycles are left effectively zero; the collate step re-zeros padded
cycles across all channels via curve_attn_mask, so no mask is needed here.
"""
import numpy as np


def add_physics_channels(curves, channels=('dqdv', 'deltaq'), ref_cycle_idx=0,
                         eps=1e-6, clip=50.0):
    curves = np.asarray(curves, dtype=np.float64)
    L, C, P = curves.shape
    assert C == 3, f"expected 3 base channels [V,I,Q], got {C}"
    half = P // 2
    segments = [(0, half), (half, P)]   # charge segment, discharge segment
    V, Q = curves[:, 0, :], curves[:, 2, :]

    extra = []
    if 'dqdv' in channels:
        extra.append(_dqdv(V, Q, segments, eps, clip))
    if 'deltaq' in channels:
        extra.append(_deltaq(V, Q, segments, ref_cycle_idx))
    if not extra:
        return curves.astype(np.float32)

    extra = np.stack(extra, axis=1)             # [L, n_new, P]
    out = np.concatenate([curves, extra], axis=1)
    return np.nan_to_num(out, nan=0.0, posinf=clip, neginf=-clip).astype(np.float32)


def _dqdv(V, Q, segments, eps, clip):
    """Per-cycle dQ/dV, computed segment-wise so charge/discharge aren't mixed."""
    L, P = V.shape
    out = np.zeros((L, P), dtype=np.float64)
    for s, e in segments:
        dV = np.gradient(V[:, s:e], axis=1)
        dQ = np.gradient(Q[:, s:e], axis=1)
        dV = np.where(np.abs(dV) < eps, eps, dV)   # guard flat-voltage regions
        out[:, s:e] = dQ / dV
    return np.clip(out, -clip, clip)


def _deltaq(V, Q, segments, ref_idx):
    """ΔQ(V): this cycle's capacity minus the reference cycle's capacity, sampled at
    this cycle's own voltages (i.e. aligned in *voltage space*, not by index)."""
    L, P = V.shape
    out = np.zeros((L, P), dtype=np.float64)
    for s, e in segments:
        Vref, Qref = V[ref_idx, s:e], Q[ref_idx, s:e]
        order = np.argsort(Vref)
        Vref_s, Qref_s = Vref[order], Qref[order]
        if len(Vref_s) < 2 or (Vref_s[-1] - Vref_s[0]) < 1e-9:
            continue                                # degenerate ref (e.g. dead/padded)
        for l in range(L):
            q_ref_at = np.interp(V[l, s:e], Vref_s, Qref_s)
            out[l, s:e] = Q[l, s:e] - q_ref_at
    return out
