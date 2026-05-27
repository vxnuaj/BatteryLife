"""Domain helpers for joint multi-domain training (Hypothesis H3).

Four domains, in id order: 0=Li-ion, 1=Zn-ion, 2=Na-ion, 3=CALB.
Li-ion is every source in MIX_large (CALCE/HNEI/HUST/MATR/...); the other three are
the single-chemistry datasets.
"""
import numpy as np

DOMAIN_NAMES = ['Li-ion', 'Zn-ion', 'Na-ion', 'CALB']


def filename_to_domain_id(filename):
    """Map a processed-cell filename to its domain id."""
    f = str(filename)
    if f.startswith('ZN-coin') or f.startswith('ZN-'):
        return 1
    if f.startswith('NA-ion') or f.startswith('NA-'):
        return 2
    if f.startswith('CALB'):
        return 3
    return 0  # everything else is a Li-ion source (MIX_large)


def per_domain_metrics(preds, targets, domain_ids, alpha=0.15):
    """Per-domain MAPE and alpha-accuracy on (un-scaled) life predictions.

    Returns {domain_name: {'mape', 'acc', 'n'}} plus an 'overall' entry, so a single
    joint test set can be scored against each domain's own SOTA.
    """
    preds = np.asarray(preds, dtype=float).reshape(-1)
    targets = np.asarray(targets, dtype=float).reshape(-1)
    domain_ids = np.asarray(domain_ids, dtype=int).reshape(-1)

    def _score(p, t):
        rel = np.abs(p - t) / t
        return {'mape': float(rel.mean()), 'acc': float((rel <= alpha).mean()), 'n': int(len(p))}

    out = {}
    for d in np.unique(domain_ids):
        m = domain_ids == d
        name = DOMAIN_NAMES[d] if 0 <= d < len(DOMAIN_NAMES) else str(d)
        out[name] = _score(preds[m], targets[m])
    out['overall'] = _score(preds, targets)
    return out
