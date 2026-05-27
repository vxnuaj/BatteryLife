# Dataset Optimization — speed & storage

The HuggingFace processed dataset is **86.4 GB**, but the benchmark uses a small fraction of
it. This documents what's actually needed (verified against the code) and the optimal way to
download/store/load it. **All of it is lossless** — model inputs and labels are unchanged.

## What the benchmark actually uses (evidence)

| Fact | Source |
|---|---|
| Labels come from `Life labels/*.json` (tiny, already shipped) — **not** computed from the pkl | `data_loader.py:480-490` (`eol = life_labels[file_name]`) |
| The loader processes only the **first 100 cycles** of each cell, then `break`s | `data_loader.py:530` (`> early_cycle_threshold or > eol`) |
| `cycle_data` is read **positionally** (`cycle_number = index+1`), so `cycle_data[:100]` is exactly what's used | `data_loader.py:519-523` |
| **SDU & Stanford_2 are never referenced** (0 of 1001 split cells) and have **no loader branch** — they can't even be loaded | split lists + `read_cell_data_according_to_prefix` has no `SDU` case; README: added *after* the KDD'25 paper (2025.10 / 2025.06) |

So: per cell we need only its **first 100 cycles** of curves + a label that's already in hand.
The benchmark's 4-domain splits reference **1001 distinct cells**.

## The numbers (HF `files_metadata`)

Total **86.4 GB / 1422 files**. Largest folders:

| folder | GB | used? |
|---|---:|---|
| ISU_ILCC | 29.65 | ✅ (all 240) |
| **Stanford_2** | **20.36** | ❌ never referenced |
| MATR | 8.32 | ✅ |
| Stanford | 4.24 | ✅ (original release) |
| Tongji | 4.17 | ✅ |
| HUST | 4.14 | ✅ |
| **SDU** | **3.96** | ❌ never referenced |
| ZN-coin | 3.19 | ✅ |
| RWTH / XJTU / MICH / NA-ion / SNL / MICH_EXP / HNEI / CALCE / CALB / UL_PUR | ≤2.6 each | ✅ |

**Unused datasets = SDU + Stanford_2 = 24.3 GB (~28%).** Slim ratio (measured on real cells,
truncating to 100 cycles, verified byte-identical for cycles ≤100): MATR 7.6×, ZN-coin 3.4×,
and ~1× for cells already ≤100 cycles (NA-ion, CALB). High-cycle folders (ISU_ILCC, MATR,
HUST, ZN-coin) shrink the most.

## The three-layer optimization (apply in order)

### 1. Don't download the unused datasets  → −24.3 GB, zero code
```bash
hf download Battery-Life/BatteryLife_Processed --repo-type dataset --local-dir dataset/ \
    --exclude "SDU/*" "Stanford_2/*"
```

### 2. Slim each cell to its first 100 cycles  → ~5–10× on high-cycle cells, lossless, drop-in
```bash
python scripts/build_slim_dataset.py --root dataset/ --dry-run   # report savings first
python scripts/build_slim_dataset.py --root dataset/             # truncate in place
```
Truncates `cycle_data` to `[:100]` (== what the loader uses). Idempotent; skips
`Life labels/`, `seen_unseen_labels/`, `READMEs/`. **No loader changes** — the existing loader
reads these slim files identically.

### 3. Cache the resampled tensors  → ~MBs per split, near-instant load (best speed)
The expensive per-run step is `read_data()`: it `pickle.load`s every cell and resamples the
≤100 cycles to `[L, 3, 300]`. That output is deterministic given
`(dataset, split, charge_discharge_length, early_cycle_threshold, input_channels)`. Pass
`--use_cache` and the **first run builds** `dataset/.cache/<key>_v1.pkl`; **every later run loads
it instantly** (no pkl reads, no resampling). Add `--use_cache` to any run script:
```bash
... run_main.py ... --use_cache
```
After the cache is built you can **delete the heavy cell pkls entirely** and run from the
(~hundreds of MB) cache alone.

## Recommended pipeline

```
download (exclude SDU/Stanford_2)   # ~62 GB
  → build_slim_dataset.py           # ~10–15 GB, lossless
  → first run with --use_cache      # builds dataset/.cache/*  (a few hundred MB total)
  → (optional) delete cell pkls     # run from cache only
```
Net: from **86 GB stored + full-file reads every run** → **a few hundred MB + instant loads**,
with identical results. You can also **stage by domain** — CALB ≈ 50 MB, Na-ion ≈ 0.7 GB,
Zn-ion ≈ 3 GB; Li-ion (ISU_ILCC+MATR+…) is ~all the bulk — so validate H1/H2/H3 on the small
domains first and pull Li-ion last.

## Why it's lossless / correct
- Slim: cycles ≤100 are byte-identical (verified); the label is read from JSON regardless; the
  loader never touches cycles >100.
- Cache: stores the **raw** (pre-scaling) `read_data()` output, so all downstream scaling
  (`--label_scaling`), weighting, and H3 domain logic run exactly as without the cache.

## Caveats
- The cache key includes `charge_discharge_length`, `early_cycle_threshold`, and
  `input_channels`; change any of those (or the resampling code) and bump the `_v1` tag /
  clear `dataset/.cache/`. It's per-split (train/val/test) and per-dataset.
- Slim assumes `early_cycle_threshold ≤ 100` (the benchmark default). To study >100 input
  cycles, re-slim with `--cycles N` (and re-download those cells, since truncation is destructive).
- `--use_cache` is **off by default**; the slim script and the cache hook were verified by
  `py_compile` + a real-data truncation test, but the cache build runs inside the (GPU-only)
  loader — smoke-run it once before relying on it.
