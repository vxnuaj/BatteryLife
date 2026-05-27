# Experiments — Beating the BatteryLife SOTA

This directory holds our research experiments aimed at beating the state-of-the-art on the
**BatteryLife (KDD 2025)** benchmark (arXiv 2502.18807). For the full project orientation
(goal, env, data, conventions) read [`../CLAUDE.md`](../CLAUDE.md); for the hypotheses'
motivation read [`../HYPOTHESES.md`](../HYPOTHESES.md); for the phased build plans see
[`../impl_plans/`](../impl_plans).

## The benchmark we're beating

Four per-domain leaderboards (**Li-ion, Zn-ion, Na-ion, CALB**), scored on **MAPE ↓** and
**15%-Acc ↑**, 6:2:2 split, 3 seeds (42/2021/2024), mean ± std.

| Domain | SOTA MAPE | SOTA holder |
|---|---|---|
| Li-ion | 0.179 | CPMLP |
| Zn-ion | 0.515 | CPTransformer |
| Na-ion | 0.255 | CPTransformer |
| CALB   | 0.140 | CPMLP |

SOTA is split between **CPMLP** and **CPTransformer** (both "CyclePatch": a shared
intra-cycle encoder per cycle → an inter-cycle encoder over the ~100 cycle tokens). No single
model wins all four domains — the opening we're targeting.

## The three experiments

| Dir | Hypothesis | Flag (default = SOTA) | CPU-verified | GPU status |
|---|---|---|---|---|
| [`h1/`](h1/) | Conv intra-cycle encoder instead of flatten→Linear | `--intra_encoder linear\|conv` | ✅ | ✅ runs as-is |
| [`h2/`](h2/) | Physics input channels `dQ/dV` + `ΔQ(V)` | `--input_channels base\|physics` (+`--enc_in 5`) | ✅ | ✅ runs as-is |
| [`h3/`](h3/) | Joint multi-domain training + FiLM domain conditioning | `--multidomain off\|on` | ✅ | ⚠️ **glue drafted, never run** |

**All three default OFF** — the single-domain SOTA path is byte-identical with no flags set.
They are designed to **compose** (e.g. `--intra_encoder conv --input_channels physics
--multidomain on` runs all three stacked).

> **H3 caveat:** the joint-training integration (domain-aware collate, per-domain label
> scalers, balanced sampling, `domain_id` threaded through the train loop + `vali_baseline`)
> is written and `py_compile`-clean but was **never executed** (no local GPU/data). Do a small
> `--multidomain on` run on a data subset before full sweeps.

## Layout of each experiment folder

```
hN/
├── EXPERIMENT.md     # full write-up: hard-to-vary hypothesis, risky predictions,
│                     #   falsification criteria, ablations, file map, reproduction, results
├── smoke_test.py     # CPU correctness check (no DeepSpeed); run: python experiments/hN/smoke_test.py
└── *_<variant>.sh    # launch scripts (conv / phys / multidomain)
```

The **core code** each experiment adds lives in shared library files (not here), documented in
each `EXPERIMENT.md` file map — e.g. `layers/intra_encoders.py` (H1),
`utils/physics_features.py` (H2), `layers/domain_conditioning.py` + `utils/domain_utils.py` (H3),
plus flag-gated edits to `models/CPMLP.py`, `models/CPTransformer.py`, `run_main.py`,
`data_provider/`, and `utils/tools.py`.

## Running an experiment (on a GPU)

> **Operator checklist with the exact setup/edit gotchas + run order:
> [`GPU_RUNBOOK.md`](GPU_RUNBOOK.md).** (Scripts ship with placeholder `checkpoints=` paths and
> a 2-GPU assumption you must edit; H3's glue needs a subset validation run first.) Summary:

1. **Set up** (GPU box): `pip install -r ../requirements.txt` (+ BatteryML per README), then
   download data: `hf download Battery-Life/BatteryLife_Processed --repo-type dataset --local-dir ../dataset/`.
2. **CPU sanity** (anywhere): `python experiments/hN/smoke_test.py` → expect `ALL CHECKS PASSED`.
3. **Reproduce the baseline first** (the un-flagged single-domain model) so we compare against
   our own numbers, not paper numbers.
4. **Run the treatment** (the `hN/*.sh` script), sweeping 3 seeds × 4 domains; record MAPE +
   15%-Acc (mean ± std).
5. **Decide** against the falsification criteria in that experiment's `EXPERIMENT.md`, and fill
   its results table.

## Methodology notes

- Each hypothesis is written to be **hard to vary** (Deutsch): the explanation is decomposed
  into parts, each tied to a breakable prediction (see the per-experiment ablations).
- **Controls:** H1 is capacity-controlled (conv encoder is *smaller* than linear, so a win is
  structural). H2 is *not* param-matched, so it includes a planned random-noise-channel control.
  H3 includes w/o-conditioning and w/o-sharing ablations.
- Keep new capability behind flags defaulting to SOTA; never change the single-domain default path.
