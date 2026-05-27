# Experiment H1 — Structure-Aware Intra-Cycle Encoder for CyclePatch

| | |
|---|---|
| **Status** | Implemented & CPU-verified (Phases 0–4). Training (Phases 5–6) pending GPU + full data. |
| **Branch** | `h1-intra-cycle-encoder` |
| **Created** | 2026-05-27 |
| **Plan** | [`../../impl_plans/H1_PLAN.md`](../../impl_plans/H1_PLAN.md) |
| **Parent doc** | [`../../HYPOTHESES.md`](../../HYPOTHESES.md) |
| **Benchmark** | BatteryLife (KDD 2025), arXiv 2502.18807 |

---

## 1. Problem context

BatteryLife predicts battery cycle-life from the first ≤100 cycles of a degradation
test. Each battery is a tensor `[100 cycles, 3 channels (V, I, Q), 300 resampled points]`.
The SOTA method, **CyclePatch**, encodes each cycle into a token (the *intra-cycle
encoder*), then models the ~100-token sequence (the *inter-cycle encoder*).

The paper establishes — through its own ablations and the failure of competing models —
a specific empirical fact: **the life-determining signal lives in the fine-grained,
time-aligned interactions among voltage, current, and capacity *within* a cycle**, not
in trend/seasonal structure. Two independent results pin this down:

- Trend/seasonal-decomposition models (DLinear, Autoformer, MICN) are weak ⇒ the signal
  is *not* in the trend/season.
- RevIN (instance normalization) *hurts* ⇒ destroying cross-variable scale relationships
  destroys signal, so the **cross-variable interaction itself carries information**.

Yet the SOTA intra-cycle encoder is the crudest possible map of that signal: it
`flatten`s the `3 × 300 = 900` numbers into one vector and applies a single dense
`Linear(900 → d_model)`.

## 2. Hypothesis (hard to vary)

> **Claim.** Replacing the intra-cycle `flatten → Linear` with a 1-D convolution over the
> 300-point curve (3 variables as input channels) lowers prediction error, because that
> convolution is the minimal architecture that *builds in* the two structural priors the
> flatten throws away — and which the paper's evidence says carry the signal.

The explanation is mechanistic and decomposed into parts, each load-bearing:

1. **The signal is local cross-variable interaction at each time region** (established §1).
2. **`flatten → Linear` is blind to two structures the data actually has:**
   - *Channel alignment.* After flattening, the value `V(t)` and the value `Q(t)` at the
     **same** timestep `t` become two arbitrary, unrelated entries of a length-900 vector.
     The linear layer has an independent weight for each, so it must **learn from data**
     that those entries co-occur at the same physical instant. Nothing in the architecture
     tells it.
   - *Translation structure.* A degradation-relevant pattern (e.g. a knee in the V–Q
     curve) appearing at charge-step 50 vs. step 52 activates entirely **different**
     weights. The linear layer must relearn the same pattern at every position.
3. **A 1-D conv with 3 input channels installs exactly — and only — those two priors:**
   each kernel application mixes V, I, Q within a *local time window at the same position*
   (channel alignment), and the kernel weights are *shared across positions* (translation
   equivariance). It adds no other assumption.

### Why this is hard to vary (Deutsch criterion)

Each component is tied to a *distinct, breakable* consequence — you cannot alter one part
without changing a prediction the experiment can check:

- Remove **cross-channel mixing** (use a *depthwise* conv: each variable convolved
  separately) → the channel-alignment prior is gone → the gain should **largely
  disappear**. If a depthwise conv recovered the full gain, the "alignment" part of the
  explanation would be false.
- Remove **weight sharing** (use a *locally-connected* layer: per-position kernels) → the
  translation prior is gone, params balloon → should help **less**, not more. If it helped
  *more*, the "translation" part would be false.
- If the signal were actually in **trend/season** (contra §1) → the conv's local prior
  would be irrelevant and decomposition models would already win. They don't.

A vague version ("CNNs beat linear layers for time series") is easy to vary and explains
nothing — it would predict conv helps the decomposition-friendly tasks too, and gives no
reason for *where* the gain should be largest. Our version forbids those outcomes.

## 3. Risky predictions

The hypothesis is only worth running because it makes predictions that could fail:

- **P1 (primary).** Conv lowers MAPE vs. linear on **Li-ion** and on **≥1 small-data domain**
  (Zn-ion / Na-ion), for **both** CPMLP and CPTransformer (shared encoder ⇒ shared lift).
- **P2 (sharp).** The **relative** gain is **larger on the small-data domains** than on
  Li-ion. Rationale tied to the mechanism: with 837 Li-ion cells the dense linear can
  *learn* the alignment from data, so the prior buys less; with 31–95 Zn/Na cells it cannot,
  so the built-in prior matters more. *If conv helps Li-ion but not the scarce domains, P2
  fails and the mechanism is suspect even if P1 holds.*
- **P3 (auxiliary, hardening).** Depthwise-conv and locally-connected ablations behave as
  in §2.3 (both worse than full conv). These are leave-one-component-out tests of the
  explanation, not just of the result.

## 4. Method

**Controlled variable — exactly one.** A single flag `--intra_encoder {linear, conv}`
selects the encoder; *everything else is identical* (inter-cycle encoder, projection,
optimizer, loss, data pipeline, normalization, seeds). `linear` is bit-for-bit the
published SOTA (it is the default), so it doubles as our reproduced baseline.

**Capacity control.** The conv encoder is deliberately *smaller* than the linear one
(36,864 vs. 115,328 params; the conv model has fewer total params). This rules out the
confound "the win came from more parameters" — any improvement is attributable to the
*structure* of the computation, not its size. (This directly supports the §2 claim and is
why the conv is kept small rather than scaled up.)

**Data contract** (`data_provider/data_loader.py:618`): `cycle_curve_data = [B, 100, 3, 300]`,
dim 2 = channels (V, I, Q), dim 3 = 300 points (150 charge + 150 discharge). Normalization:
V ÷ max(V), I ÷ Q_nominal, Q ÷ Q_nominal.

**Architecture (conv path).** `[B,100,3,300] → reshape [B·100,3,300] → Conv1d(3→16,k7) →
ReLU → Conv1d(16→32,k7) → ReLU → AdaptiveAvgPool1d(8) → Linear(256→d_model) →
[B,100,d_model]`. Downstream CyclePatch stack unchanged.

**Datasets / domains.** Li-ion, Zn-ion, Na-ion, CALB (the 4 benchmark leaderboards).

**Metrics.** MAPE (↓, primary) and 15%-Acc (↑), per domain.

**Protocol.** Benchmark-exact: 6:2:2 random split, Adam, **3 seeds (42, 2021, 2024)**,
report **mean ± std**. Compare conv vs. the reproduced `linear` baseline in *our* harness,
never against paper numbers.

## 5. Falsification criteria

- **H1 is refuted** if conv does **not** improve MAPE beyond noise (mean ± std overlap) on
  Li-ion **and** at least one small domain (P1 fails).
- **The mechanism is undermined** (even if P1 holds) if P2 fails (no larger relative gain on
  scarce domains) or P3 fails (ablations don't degrade as predicted).

## 6. File map

### Experiment artifacts (this folder, `experiments/h1/`)
| File | Role |
|---|---|
| `EXPERIMENT.md` | This write-up. |
| `smoke_test.py` | Standalone CPU correctness harness (no DeepSpeed). Builds both variants of both models; checks output shape, finite loss, backprop, and the param budget. **Status: PASSED.** |
| `CPMLP_conv.sh` | Launch CPMLP with `--intra_encoder conv` (renamed ids so checkpoints don't collide with baseline). |
| `CPTransformer_conv.sh` | Same for CPTransformer. |

### In-place source changes (cannot be "moved" — they are edits to shared library files)
| File | Change |
|---|---|
| `layers/intra_encoders.py` | **New.** `ConvIntraEncoder` — the core H1 module. Lives in `layers/` because the model package imports it (`from layers.intra_encoders import ConvIntraEncoder`); relocating a shared layer into `experiments/` would couple model code to an experiment folder. |
| `run_main.py` | Added `--intra_encoder {linear, conv}` arg (default `linear`). |
| `models/CPMLP.py` | `__init__`/`forward` branch on `intra_encoder`; original code preserved in the `else:` (linear) path. |
| `models/CPTransformer.py` | Same branching. |

### Baselines (unmodified, in `train_eval_scripts/`)
`CPMLP.sh`, `CPTransformer.sh` — run with default `--intra_encoder linear` to produce the
baseline arm of the A/B.

## 7. Reproduction

**Now (CPU, no data needed) — correctness:**
```bash
python experiments/h1/smoke_test.py        # expect: ALL CHECKS PASSED
```

**Later (GPU + full dataset) — the actual experiment:**
```bash
# baseline arm (linear): the original scripts
sh train_eval_scripts/CPMLP.sh
sh train_eval_scripts/CPTransformer.sh
# treatment arm (conv):
sh experiments/h1/CPMLP_conv.sh
sh experiments/h1/CPTransformer_conv.sh
# sweep all 3 seeds (42,2021,2024) × 4 domains for each; record MAPE / 15%-Acc.
```
A multi-seed/domain sweep wrapper will be added in the GPU phase.

## 8. Results

*Pending GPU runs. Fill mean ± std over 3 seeds; bold the better of {linear, conv} per cell.*

| Backbone | Encoder | Li-ion MAPE | Li-ion 15%-Acc | Zn-ion MAPE | Na-ion MAPE | CALB MAPE |
|---|---|---|---|---|---|---|
| CPMLP | linear (baseline) | — | — | — | — | — |
| CPMLP | conv (H1) | — | — | — | — | — |
| CPTransformer | linear (baseline) | — | — | — | — | — |
| CPTransformer | conv (H1) | — | — | — | — | — |

**Reference SOTA targets** (paper): Li-ion 0.179 / Zn-ion 0.515 / Na-ion 0.255 / CALB 0.140 (MAPE).

## 9. Threats to validity

- **Hyperparameters tuned for the linear encoder** may disadvantage conv; mitigate by reusing
  the baseline's tuned settings *and* a small conv-only sanity sweep before declaring failure.
- **Conv hyperparameters (kernel, channels, pool)** are a free axis; we fix sane defaults and
  test the *structural* claim first, tuning only if the signal is promising (avoids
  garden-of-forking-paths).
- **Run-to-run variance** on tiny domains (Zn/Na) is large (paper std up to ±0.25 on 15%-Acc);
  3 seeds may be too few to resolve small effects — may need more seeds for P2.

## 10. Changelog
- 2026-05-27 — Phases 0–4 implemented; smoke test passing; artifacts moved to `experiments/h1/`; write-up created.
