# CLAUDE.md — Project Context & Onboarding

> Read this first. It orients any coding agent (or human) to the in-progress research on
> this fork. The detailed records live in `HYPOTHESES.md`, `impl_plans/`, and `experiments/`.

## Goal

Beat the **state-of-the-art on the BatteryLife (KDD 2025) benchmark** for battery
life prediction (arXiv 2502.18807). The benchmark = 4 per-domain leaderboards
(**Li-ion, Zn-ion, Na-ion, CALB**), each scored on **MAPE ↓** and **15%-Acc ↑**, with a
6:2:2 split, **3 seeds (42, 2021, 2024)**, mean ± std.

**SOTA to beat (paper, MAPE):** Li-ion 0.179 · Zn-ion 0.515 · Na-ion 0.255 · CALB 0.140.
SOTA is split between the paper's own models **CPMLP** and **CPTransformer** (both use
"CyclePatch": a shared intra-cycle encoder per cycle, then an inter-cycle encoder over the
~100 cycle tokens). No single model wins all four domains — that gap is the opening.

## The three hypotheses (full rationale in `HYPOTHESES.md`)

| | Idea | Flag (default = SOTA) | Where |
|---|---|---|---|
| **H1** | Structure-aware **conv** intra-cycle encoder instead of flatten→Linear | `--intra_encoder linear\|conv` | `layers/intra_encoders.py`, `experiments/h1/` |
| **H2** | **Physics** input channels: `dQ/dV` + `ΔQ(V)` on top of raw V/I/Q | `--input_channels base\|physics` (+`--enc_in 5`) | `utils/physics_features.py`, `experiments/h2/` |
| **H3** | **Joint multi-domain** training + FiLM domain conditioning | `--multidomain off\|on` (+`--num_domains`, `--label_scaling`) | `layers/domain_conditioning.py`, `utils/domain_utils.py`, `experiments/h3/` |

All three are **off by default** — the single-domain SOTA path is byte-identical when no flag
is set. They are designed to **compose** (conv encoder + physics channels + domain conditioning).
Each `experiments/h{1,2,3}/EXPERIMENT.md` is a full write-up (hard-to-vary hypothesis, risky
predictions, falsification criteria, ablations, file map, reproduction).

## Status (as of 2026-05-27) — IMPORTANT

| Exp | CPU-verified | GPU-ready? |
|---|---|---|
| H1 | ✅ smoke test passes | ✅ runs as-is (`--intra_encoder conv` routes through the unmodified loop) |
| H2 | ✅ smoke test (incl. real-data feature check) | ✅ runs as-is |
| H3 | ✅ smoke test passes | ⚠️ **GPU glue is DRAFTED but never executed** — see below |

**H3 caveat (read before running):** the joint-training integration (domain-aware collate,
per-domain label scalers, balanced sampling, `domain_id` threaded through the train loop and
`vali_baseline`, per-domain metric logging) is written, `py_compile`-clean, and guarded
default-off — but it was **never run** (no GPU/data locally; needs `batteryml`/`denseweight`/
`sklearn`). **Do a tiny `--multidomain on` run on a data subset first** to shake it out before
full sweeps.

Nothing is merged to `main`; all work is on feature branches (mainly `h3-multidomain`, which
contains the cumulative H1+H2+H3 changes).

## Repo orientation

- `HYPOTHESES.md` — the 3 hypotheses, full motivation, what the data analysis showed.
- `impl_plans/H{1,2,3}_PLAN.md` — phased implementation plans with checklists (phase status marked).
- `experiments/h{1,2,3}/` — per-experiment: `EXPERIMENT.md` (write-up), `smoke_test.py` (CPU
  correctness, no DeepSpeed), and launch scripts (`*_conv.sh` / `*_phys.sh` / `*_multidomain.sh`).
- Core code: `run_main.py` (train entry), `models/CPMLP.py` & `models/CPTransformer.py`
  (the SOTA models, edited for all 3 flags), `data_provider/` (loaders, splits),
  `layers/`, `utils/`.

## Environment & data

- **Compute:** training is hardwired to DeepSpeed (`run_main.py`) → needs an NVIDIA GPU. Models
  are tiny (CPMLP ~12M, CPTransformer ~1M); a single consumer GPU suffices.
- **Install:** `pip install -r requirements.txt` (also install BatteryML per README). The CPU
  smoke tests only need `torch numpy` (+`reformer_pytorch` for CPTransformer import).
- **Data (NOT in git):** gated HF dataset `Battery-Life/BatteryLife_Processed` (~86 GB; pkls
  store full-resolution cycling data). Download onto the GPU box:
  ```
  hf download Battery-Life/BatteryLife_Processed --repo-type dataset --local-dir dataset/
  ```
  (Requires `hf auth login` + accepting the dataset terms on HF. `dataset/` is gitignored
  except the tracked `seen_unseen_labels/`.) A ~105 MB subset was used for local dev.

## How to run (on GPU)

CPU correctness check (anywhere): `python experiments/h{1,2,3}/smoke_test.py`

Per experiment — **reproduce the `linear`/`base`/single-domain baseline first**, then the treatment:
```bash
# H1:  sh train_eval_scripts/CPMLP.sh        (baseline) ; sh experiments/h1/CPMLP_conv.sh
# H2:  sh train_eval_scripts/CPMLP.sh        (baseline) ; sh experiments/h2/CPMLP_phys.sh
# H3:  sh experiments/h3/CPMLP_multidomain.sh  (dataset=MIX_all / MIX_all42 / MIX_all2024 per seed)
```
Sweep 3 seeds × 4 domains; record MAPE + 15%-Acc (mean ± std); compare to the reproduced
baseline, never to paper numbers. Composition runs (e.g. `--intra_encoder conv --input_channels
physics --multidomain on`) test stacked hypotheses.

## Working conventions

- Keep every new capability behind a flag that defaults to the SOTA behaviour; never change the
  single-domain default path.
- Each experiment has a `smoke_test.py` — extend/run it before trusting a change.
- Log results back into the relevant `experiments/h*/EXPERIMENT.md` (results table + go/no-go vs
  the falsification criteria stated there).
