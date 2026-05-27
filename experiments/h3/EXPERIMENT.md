# Experiment H3 — Joint Multi-Domain Training with Domain Conditioning

| | |
|---|---|
| **Status** | Fully drafted & CPU-verified (Phases 0–6 done; train-loop/loader/eval glue drafted + compile-clean, guarded default-off). Training (7,8) + a GPU smoke-run of the glue pending GPU + full data. |
| **Branch** | `h3-multidomain` |
| **Created** | 2026-05-27 |
| **Plan** | [`../../impl_plans/H3_PLAN.md`](../../impl_plans/H3_PLAN.md) |
| **Parent doc** | [`../../HYPOTHESES.md`](../../HYPOTHESES.md) |
| **Benchmark** | BatteryLife (KDD 2025), arXiv 2502.18807 |

---

## 1. Problem context

The benchmark trains and evaluates **one model per domain** (Li-ion, Zn-ion, Na-ion, CALB).
The scarce domains are weak purely from data starvation: Zn-ion has 95 cells across 45
chemical systems (~2 cells/system), Na-ion has 31 cells, CALB 27 — versus 837 Li-ion cells.
The paper tried to share strength via **sequential** transfer from Li-ion (frozen features,
fine-tuning, domain adaptation) and reported all three **underperform** training from scratch.
It never tried **joint** training, where all domains are learned at once.

## 2. Hypothesis (hard to vary)

> **Claim.** A single model trained jointly on all four domains, with a **shared intra-cycle
> encoder** and a **FiLM domain-conditioned** inter-cycle path, beats the per-domain SOTA on
> the scarce domains (Zn-ion, Na-ion) **without** degrading Li-ion.

Decomposed into load-bearing parts:

1. **Within-cycle electrochemical structure is partially shared across chemistries.** The
   shape of V/I/Q within a cycle obeys common physics, so a shared encoder trained mostly on
   the 837 Li-ion cells learns reusable low-level features that 31–95-cell domains cannot
   learn alone. → the **pooling** mechanism.
2. **Between-domain distribution differs** (life scale ~7×; Q(V) shape per chemistry), so
   forcing one weight set to fit all causes negative transfer. **FiLM** supplies the
   *minimal* per-domain freedom (feature-wise scale/shift) to specialise without separate
   networks. → the **conditioning** mechanism.
3. **Joint co-adaptation explains the paper's transfer failure.** Frozen features can't adapt
   to a new chemistry; full fine-tuning forgets the source / overfits a tiny target. Training
   shared + domain-specific parameters *simultaneously* avoids both.

### Why this is hard to vary (Deutsch criterion)

Each part has a dedicated ablation that breaks a *specific* prediction:

- Remove **conditioning** (pool all domains, no FiLM): one forced mapping ⇒ Li-ion should
  regress (negative transfer) and/or scarce domains shouldn't specialise. If unconditioned
  joint training matched conditioned, part 2 is false.
- Remove **sharing** (per-domain encoders): collapses to separate models ⇒ scarce domains
  should **not** gain. If they still gained, part 1 (pooling) isn't the mechanism.
- **Global vs per-domain label scaler**: a single global scaler squashes small-life domains;
  if global matched per-domain, the scale-shift account (part 2) is incomplete.

A vague "training on more data helps" is easy to vary and predicts gains regardless of
conditioning, sharing, or scaling — exactly what these ablations are designed to forbid.

## 3. Risky predictions

- **P1 (primary).** Zn-ion and Na-ion MAPE improve vs. per-domain SOTA (both backbones).
- **P2.** Li-ion is **not** materially hurt (within noise) — no net negative transfer.
- **P3 (ablation).** *w/o conditioning* < *w/ conditioning* (Li-ion regresses or scarce gain vanishes).
- **P4 (ablation).** *w/o shared encoder* erases the scarce-domain gain (≈ per-domain models).
- **P5.** Domain-balanced sampling is necessary for the scarce-domain gain (else Li-ion swamps).

## 4. Method

**Controlled variable.** A flag `--multidomain {off, on}` (default `off`). `off` is bit-for-bit
single-domain SOTA. Conditioning modules are built/active only when `on` **and** a `domain_id`
is supplied; `domain_id=None` ⇒ identity, so the default path is provably unchanged.

**Conditioning = FiLM** (`layers/domain_conditioning.py`): a per-domain embedding →
`(gamma, beta)` applied as `x*(1+gamma)+beta` to the `[B, S, d_model]` cycle-token embeddings,
between the (shared) intra-cycle encoder and the inter-cycle encoder. **Zero-initialized**, so
it starts as the identity and learns specialization from the strong base model. Alternatives to
ablate: `concat` embedding, per-domain `heads` (`--domain_cond`).

**Shared encoder.** The intra-cycle encoder (linear or H1-conv) is fully shared — the pooling
mechanism — so H3 composes with H1/H2.

**Label scaling.** `--label_scaling per_domain` (one StandardScaler per domain) is primary;
`global` and `log` are ablations. (Per-domain because life medians span ~170→1171; a global
scaler squashes the scarce domains we aim to help.)

**Data.** Joint split `MIX_all{,42,2024}` (= `MIX_large` + seed-matched Zn/CALB/Na). Per-sample
`domain_id ∈ {0:Li-ion, 1:Zn, 2:Na, 3:CALB}` via `filename_to_domain_id`.

**Evaluation.** Per-domain MAPE / 15%-Acc on the joint test set (`per_domain_metrics`), each
inverse-transformed with its domain scaler — so results compare directly to per-domain SOTA.

**Sampling.** Domain-balanced (inverse-frequency) via `--weighted_sampling`, so Li-ion (837)
doesn't swamp Na-ion (31).

**Protocol.** 3 seeds (via the 3 `MIX_all` splits), mean ± std; compare to reproduced per-domain SOTA.

## 5. Falsification criteria

- **H3 refuted** if Zn-ion/Na-ion do **not** improve vs per-domain SOTA, **or** if Li-ion
  regresses materially (negative transfer dominates).
- **Mechanism undermined** if P3 fails (conditioning doesn't matter) or P4 fails (sharing doesn't matter).

## 6. File map

### Experiment artifacts (`experiments/h3/`)
| File | Role |
|---|---|
| `EXPERIMENT.md` | This write-up. |
| `smoke_test.py` | CPU harness: FiLM (identity@init + domain-sensitivity), utils, combined-split coverage, CP{MLP,Transformer} on/off-path. **PASSED.** |
| `CPMLP_multidomain.sh`, `CPTransformer_multidomain.sh` | Joint launch (`--dataset MIX_all --multidomain on --num_domains 4 --label_scaling per_domain --weighted_sampling`). |

### In-place source changes (shared files)
| File | Change |
|---|---|
| `layers/domain_conditioning.py` | **New.** `FiLMConditioner`. |
| `utils/domain_utils.py` | **New.** `filename_to_domain_id`, `per_domain_metrics`, `DOMAIN_NAMES`. |
| `run_main.py` | Added `--multidomain`, `--domain_cond`, `--num_domains`, `--label_scaling`. |
| `data_provider/data_loader.py` | Added `MIX_all` / `MIX_all42` / `MIX_all2024` dataset branches. |
| `models/CPMLP.py`, `models/CPTransformer.py` | `forward(..., domain_id=None)` + FiLM on cycle tokens when `multidomain on`. |

### GPU-phase glue — now DRAFTED (compile-clean, guarded default-off, not yet run)
All written without changing the default path (a **new** `my_collate_fn_multidomain` is used
only under `--multidomain on`; every loop/eval branch is flag-guarded; non-CP models never
receive the `domain_id` kwarg). Needs a GPU smoke-run before trust:
- `data_provider/data_loader.py` — `domain_id` per sample (`read_data` → `__getitem__`),
  `my_collate_fn_multidomain` (8-tuple), **per-domain `StandardScaler`s** under `--label_scaling per_domain`.
- `data_provider/data_factory.py` — selects the multidomain collate + a `WeightedRandomSampler`
  (inverse domain frequency) on train when `--multidomain on --weighted_sampling`.
- `run_main.py` — train loop unpacks/passes `domain_id`, per-domain inverse-transform, stashes
  train domain stats on `args` for val/test.
- `utils/tools.py:vali_baseline` — guarded 8-tuple unpack, per-domain inverse-transform, gathers
  `domain_id`, logs per-domain MAPE/Acc (return signature unchanged).

**Caveat:** verified by `py_compile` + the CPU smoke test + inspection only; the joint loader
and training loop have **not been executed** (needs GPU + full data + `batteryml`/`denseweight`/`sklearn`).

## 7. Reproduction

**Now (CPU) — correctness:**
```bash
python experiments/h3/smoke_test.py        # expect: ALL CHECKS PASSED
```

**Later (GPU) — after the deferred threading is wired:**
```bash
sh experiments/h3/CPMLP_multidomain.sh          # seed 2021 (MIX_all)
# edit dataset=MIX_all42 / MIX_all2024 + seed for the other two seeds; repeat for CPTransformer.
# ablations: drop --multidomain (off), drop --weighted_sampling, --label_scaling global,
#            and a per-domain-encoder variant.
```

## 8. Results

*Pending GPU runs. Per domain, mean ± std over 3 seeds; compare to per-domain SOTA.*

| Backbone | Setting | Li-ion MAPE | Zn-ion MAPE | Zn-ion 15%-Acc | Na-ion MAPE | Na-ion 15%-Acc | CALB MAPE |
|---|---|---|---|---|---|---|---|
| CPMLP | per-domain SOTA | — | — | — | — | — | — |
| CPMLP | joint + FiLM | — | — | — | — | — | — |
| CPTransformer | per-domain SOTA | — | — | — | — | — | — |
| CPTransformer | joint + FiLM | — | — | — | — | — | — |

**Reference SOTA (paper):** Li-ion 0.179 / Zn-ion 0.515 / Na-ion 0.255 / CALB 0.140 (MAPE).

## 9. Threats to validity

- **"More data" confound** — joint training sees more cells. The **w/o-shared-encoder** ablation
  (P4) controls this: if pooling isn't the mechanism, separate encoders should match the joint model.
- **Label-scaling sensitivity** — per-domain vs global is a large lever (and an ablation).
- **FiLM may be too weak** (tiny capacity); `concat`/`heads` are fallbacks via `--domain_cond`.
- **Negative transfer** to Li-ion is a real failure mode (P2 guards against it).
- Small-domain variance is large; P1/P3/P4 may need >3 seeds to resolve.

## 10. Changelog
- 2026-05-27 — Phases 0,2,5,6 implemented; 1,4 partial; smoke test passing; H1/H2 unaffected; write-up created. Train-loop/loader threading deferred to GPU phase.
