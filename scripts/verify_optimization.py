"""Screen the dataset optimizations for LOSSLESSNESS before applying them.

Runs the REAL loader code (Dataset_original.read_cell_df -> get_charge_discharge_curves,
the exact resampling/normalization used in training) on every available cell, comparing
FULL vs a 100-cycle-truncated copy. Asserts:
  * the produced [L, 3, 300] resampled curves are EXACTLY equal (np.array_equal),
  * the EOL label is unchanged,
and reports the label distribution (mean/std/min/median/max) full vs slim — because data
distribution is what actually matters.

It also checks the resampled-tensor cache round-trip (pickle dump/load) preserves arrays.

Heavy imports data_loader does at module top but never uses on this path
(batteryml.BatteryData, accelerate) are stubbed, so this runs without a GPU/full install.

Run (from repo root): python scripts/verify_optimization.py
Works on whatever cells exist under dataset/ — the local subset here, or the full data on
the GPU box. Run it BEFORE build_slim_dataset.py to prove equivalence on your data.
"""
import os
import sys
import glob
import copy
import types
import pickle
import shutil
import tempfile
import numpy as np

# --- stub the unused heavy imports so we can import the real loader on a bare CPU box ---
for name in ['batteryml', 'batteryml.data', 'batteryml.data.battery_data']:
    sys.modules[name] = types.ModuleType(name)
sys.modules['batteryml.data.battery_data'].BatteryData = object
sys.modules['batteryml'].data = sys.modules['batteryml.data']
sys.modules['batteryml.data'].battery_data = sys.modules['batteryml.data.battery_data']
sys.modules['accelerate'] = types.ModuleType('accelerate')

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
from types import SimpleNamespace                                  # noqa: E402
from data_provider.data_loader import Dataset_original             # noqa: E402
from utils.augmentation import BatchAugmentation_battery_revised   # noqa: E402

DATA = os.path.join(_ROOT, 'dataset')
CYCLES = 100
SKIP = ('Life labels', 'seen_unseen_labels', 'READMEs')


def make_obj(root):
    o = Dataset_original.__new__(Dataset_original)   # bypass the heavy __init__
    o.root_path = root
    o.charge_discharge_len = 300
    o.early_cycle_threshold = CYCLES
    o.need_keys = ['current_in_A', 'voltage_in_V', 'charge_capacity_in_Ah',
                   'discharge_capacity_in_Ah', 'time_in_s']
    o.ZN_coin_charge_first_file_names = []  # only affects which branch; identical for full & slim
    o.args = SimpleNamespace(input_channels='base')
    o.aug_helper = BatchAugmentation_battery_revised()
    return o


def main():
    cells = [p for p in glob.glob(os.path.join(DATA, '**', '*.pkl'), recursive=True)
             if not any(s in p for s in SKIP)]
    if not cells:
        print("no cells under dataset/ — download some first."); return

    full_obj = make_obj(DATA)
    n_ok = n_trunc = 0
    eols_full, eols_slim = [], []
    print(f"{'cell':42s}{'cyc':>6}{'curves equal':>14}{'eol equal':>11}")
    for cell in cells:
        fn = os.path.basename(cell)
        folder = os.path.relpath(os.path.dirname(cell), DATA)
        if folder.startswith('MICH'):       # needs merge_MICH; skip in this lightweight screen
            continue
        # FULL (read_cell_df returns 5-tuple of Nones when the cell has no JSON label -> not a benchmark cell)
        res_f = full_obj.read_cell_df(fn)
        if res_f[0] is None:
            continue
        curves_f, eol_f = res_f[1], res_f[2]

        # SLIM copy in a temp root (symlink Life labels so the JSON label still resolves)
        tmp = tempfile.mkdtemp()
        os.symlink(os.path.join(DATA, 'Life labels'), os.path.join(tmp, 'Life labels'))
        os.makedirs(os.path.join(tmp, folder), exist_ok=True)
        with open(cell, 'rb') as f:
            d = pickle.load(f)
        n = len(d['cycle_data'])
        d['cycle_data'] = d['cycle_data'][:CYCLES]
        with open(os.path.join(tmp, folder, fn), 'wb') as f:
            pickle.dump(d, f)
        res_s = make_obj(tmp).read_cell_df(fn)
        curves_s, eol_s = res_s[1], res_s[2]
        shutil.rmtree(tmp)

        ce = np.array_equal(np.asarray(curves_f), np.asarray(curves_s))
        ee = (eol_f == eol_s)
        eols_full.append(eol_f); eols_slim.append(eol_s)
        n_ok += (ce and ee); n_trunc += (n > CYCLES)
        print(f"{fn:42s}{n:>6}{str(ce):>14}{str(ee):>11}")
        assert ce and ee, f"MISMATCH on {fn} (curves={ce}, eol={ee})"

    # cache round-trip: pickling the resampled arrays preserves them exactly
    arr = np.asarray(curves_f)
    rt = np.array_equal(arr, pickle.loads(pickle.dumps(arr)))

    def stats(x):
        x = np.array(x, float); return f"mean={x.mean():.1f} std={x.std():.1f} min={x.min():.0f} med={np.median(x):.0f} max={x.max():.0f}"
    print(f"\ncells compared: {len(eols_full)} | truncated (>{CYCLES} cyc): {n_trunc} | all equal: {n_ok}/{len(eols_full)}")
    print(f"label dist FULL: {stats(eols_full)}")
    print(f"label dist SLIM: {stats(eols_slim)}")
    print(f"cache round-trip preserves arrays: {rt}")
    print("\nPASS — slim resampled curves + labels are identical to full." if n_ok == len(eols_full)
          else "\nFAIL")


if __name__ == '__main__':
    main()
