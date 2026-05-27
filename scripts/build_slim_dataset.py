"""Slim the BatteryLife processed dataset to what the benchmark actually uses.

WHY (verified against the code):
  * The loader reads each cell's label from `Life labels/*.json`, NOT from the pkl
    (data_loader.py:480-490) -- those JSONs are tiny and already present.
  * The loader only ever processes the first `early_cycle_threshold` (=100) cycles of a
    cell, `break`-ing after that (data_loader.py:530). Every cycle >100 is loaded into
    RAM and immediately discarded.
So truncating each cell's `cycle_data` to its first N (=100) cycles is LOSSLESS for the
benchmark (cycles <=N are byte-identical, the label is unchanged) and cuts storage +
per-run load I/O substantially (up to ~8x on high-cycle cells like MATR/ISU_ILCC/ZN-coin).

PAIR THIS WITH skipping the datasets the benchmark never references (24.3 GB, ~28%):
    hf download Battery-Life/BatteryLife_Processed --repo-type dataset --local-dir dataset/ \
        --exclude "SDU/*" "Stanford_2/*"

USAGE:
    python scripts/build_slim_dataset.py --root dataset/ --dry-run   # report only
    python scripts/build_slim_dataset.py --root dataset/             # truncate in place

Idempotent (re-running on already-slim files is a no-op). Does not touch
`Life labels/`, `seen_unseen_labels/`, or `READMEs/`.
"""
import argparse
import glob
import io
import os
import pickle

SKIP_DIRS = ('Life labels', 'seen_unseen_labels', 'READMEs')


def slim_size_bytes(obj):
    buf = io.BytesIO()
    pickle.dump(obj, buf)
    return len(buf.getvalue())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='dataset', help='dataset root to slim in place')
    ap.add_argument('--cycles', type=int, default=100, help='cycles to keep (= early_cycle_threshold)')
    ap.add_argument('--dry-run', action='store_true', help='report savings without writing')
    args = ap.parse_args()

    pkls = [p for p in glob.glob(os.path.join(args.root, '**', '*.pkl'), recursive=True)
            if not any(s in p for s in SKIP_DIRS)]
    full_total = slim_total = 0
    n_trunc = 0
    for p in pkls:
        fsz = os.path.getsize(p)
        full_total += fsz
        with open(p, 'rb') as f:
            d = pickle.load(f)
        cd = d.get('cycle_data')
        if cd is None or len(cd) <= args.cycles:
            slim_total += fsz
            continue
        d['cycle_data'] = cd[:args.cycles]            # positional, == loader's break@N
        n_trunc += 1
        if args.dry_run:
            slim_total += slim_size_bytes(d)
        else:
            with open(p, 'wb') as f:
                pickle.dump(d, f)
            slim_total += os.path.getsize(p)

    mode = 'DRY-RUN (nothing written)' if args.dry_run else 'truncated in place'
    print(f"[{mode}] {len(pkls)} pkls scanned | {n_trunc} truncated to first {args.cycles} cycles")
    print(f"  {full_total/1e9:.2f} GB -> {slim_total/1e9:.2f} GB "
          f"({(1 - slim_total/max(full_total,1))*100:.0f}% smaller)")


if __name__ == '__main__':
    main()
