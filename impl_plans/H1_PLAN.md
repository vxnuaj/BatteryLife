# H1 Implementation Plan — Structure-Aware Intra-Cycle Encoder

> **Hypothesis (H1):** Replacing the intra-cycle encoder's `flatten → Linear` step
> with a structure-aware encoder (a small 1-D CNN over the `[3, 300]` cycle signal)
> lowers MAPE across domains. Because the intra-cycle encoder is **shared by the whole
> CyclePatch family**, the same change should lift **CPMLP** and **CPTransformer**
> simultaneously. See `HYPOTHESES.md` for full motivation.

## Design decision (drives the whole plan)

We implement the encoder swap as a **single config flag** —
`--intra_encoder {linear, conv}` — inside the existing `CPMLP.py` and
`CPTransformer.py`, rather than as forked model files. This makes H1 a **clean
controlled ablation**: identical everything else, one variable changed. `linear`
reproduces today's SOTA exactly (default → zero regression risk); `conv` is H1.

**Current intra path (the thing we replace):**
`X_i:[B,100,3,300] → flatten → [B,100,900] → Linear(900→D) → L×MLPBlock`
**H1 conv path:**
`X_i:[B,100,3,300] → per-cycle Conv1d stack over 300 pts, 3 in-channels → [B,100,D] → L×MLPBlock`
The inter-cycle encoder, projection, loss, and data pipeline are **untouched**.

## Hard constraints to respect

- **No GPU locally + DeepSpeed is hardwired** (`run_main.py`). Full training runs on a
  CUDA machine later. Locally we can only do a **CPU correctness smoke test** that
  bypasses DeepSpeed.
- **Only a data subset is downloaded** (7 pkls + labels). Smoke tests use the subset or
  synthetic tensors; real training needs the full 86 GB on the GPU box.
- **Must compare against a reproduced baseline**, never paper numbers. `--intra_encoder
  linear` IS that baseline, run in our own harness.
- Don't break the existing `linear` path — it must stay bit-for-bit the default.

---

## Phase 0 — Prerequisites & guardrails  ✅ DONE
*Set up so the rest is mechanical and reversible.*

- [x] Create a git branch for H1 work (`h1-intra-cycle-encoder`).
- [x] Confirm the data tensor contract from `data_provider/data_loader.py`:
      `cycle_curve_data = [B, 100, 3, 300]` — dim2 = channels (V, I, Q), dim3 = resampled
      length (300 = 150 charge + 150 discharge). Per-cycle build at `data_loader.py:618-626`.
      Normalization: V÷max(V), I÷Q_nominal, Q÷Q_nominal. `curve_attn_mask = [B, 100]`.
- [x] Reference config noted (`CPMLP.sh`/`CPTransformer.sh`): d_model=128, d_ff=256,
      e_layers 4/6, d_layers 2/4, charge_discharge_length=300, early_cycle_threshold=100.
- [x] Param-budget rule fixed: conv encoder must be **≤ 115,328** (the linear
      `intra_embed = Linear(900→128)`). Achieved: conv = 36,864.

## Phase 1 — Build the conv intra-cycle encoder (the core of H1)  ✅ DONE
*The one new piece of modeling code.*

- [x] Added `layers/intra_encoders.py` with `ConvIntraEncoder`: `[B,S,3,300]` →
      reshape `[B*S,3,300]` → Conv1d stack (3→16→32, k=7) → AdaptiveAvgPool1d(8) →
      `Linear(256→d_model)` → `[B,S,d_model]`.
- [x] Small + configurable (hidden_channels, kernel_size, pool_len, dropout). 36,864 params
      at d_model=128, under the 115,328 budget.
- [x] Unit-tested in isolation (in `experiments/h1_smoke_test.py`): `[2,100,3,300]` →
      `[2,100,128]`, finite, gradients flow.

## Phase 2 — Wire the flag into the SOTA models  ✅ DONE
*Make the swap a one-line config change, default = current SOTA.*

- [x] Added `--intra_encoder {linear, conv}` arg in `run_main.py` (default `linear`).
- [x] Branched `models/CPMLP.py` and `models/CPTransformer.py` on
      `getattr(configs,'intra_encoder','linear')` — `linear` keeps `intra_flatten +
      intra_embed` as-is; `conv` uses `ConvIntraEncoder`. Downstream unchanged.
- [x] `linear` path unchanged: original code lives in the `else:` branch and the
      `getattr` default is `linear`, so existing scripts/checkpoints behave identically.

## Phase 3 — Run-script & config plumbing  ✅ DONE
*Make the experiment launchable and reproducible.*

- [x] Created `train_eval_scripts/CPMLP_conv.sh` and `CPTransformer_conv.sh` (copies of the
      originals + `--intra_encoder conv`, with `model_id`/`comment` renamed `*_conv` so
      checkpoints/logs don't collide with the linear baseline).
- [x] Originals (`CPMLP.sh`, `CPTransformer.sh`) serve as the `linear` baseline for the A/B.
- [x] 3-seed × 4-domain coverage is expressible via the existing `--seed` and `--dataset`
      args (seeds 42/2021/2024; datasets per-domain). Sweep wrapper deferred to Phase 5.

## Phase 4 — Local CPU smoke test (correctness, not performance)  ✅ DONE
*Prove it runs end-to-end before burning GPU time.*

- [x] Wrote `experiments/h1_smoke_test.py` (standalone, no DeepSpeed). Both variants of
      CPMLP and CPTransformer: output `[4,1]`, finite loss, `backward()` + finite grads. PASS.
- [x] Param counts printed & budget enforced: conv intra = 36,864 ≤ linear intra = 115,328;
      conv total < linear total for both models (structure, not capacity).
- [x] Conv variant accepts the real `[B,100,3,300]` contract — verified with synthetic
      tensors matching `data_loader.py`'s shape/axis order (incl. padded-cycle masking).
      *Note:* used synthetic rather than piping subset pkls through the full loader, since
      the loader needs the split-recorder + complete label setup; deferred to Phase 5 on GPU.

## Phase 5 — Full training runs (GPU machine)
*The actual experiment. Deferred until GPU + full data are available.*

- [ ] On the CUDA box: download full dataset, install `requirements.txt`, reproduce
      baseline `--intra_encoder linear` for CPMLP & CPTransformer (sanity vs paper).
- [ ] Run the A/B: `{linear, conv} × {CPMLP, CPTransformer} × {4 domains} × {3 seeds}`.
- [ ] Log MAPE + 15%-Acc per run (wandb/CSV); capture mean ± std.

## Phase 6 — Analysis & go/no-go decision
*Decide whether H1 is confirmed.*

- [ ] Build the results table: conv vs linear, per domain, mean ± std, for both backbones.
- [ ] **Falsification check (from HYPOTHESES.md):** H1 fails if conv does **not** improve
      MAPE (beyond noise) on at least **Li-ion and one small domain**.
- [ ] Record outcome in `HYPOTHESES.md` (mark H1 confirmed/refuted) and memory.
- [ ] If confirmed → conv intra-encoder becomes the new shared base for H2/H3.

---

## Deliverables
- `layers/intra_encoders.py` — `ConvIntraEncoder`
- edits to `models/CPMLP.py`, `models/CPTransformer.py`, `run_main.py`
- `train_eval_scripts/CPMLP_conv.sh`, `CPTransformer_conv.sh`
- `experiments/h1_smoke_test.py`
- results table + decision logged back into `HYPOTHESES.md`

## Out of scope (explicitly)
- H2 (physics features), H3 (multi-domain), H4 (log-target) — separate plans.
- Hyperparameter search for the conv encoder beyond a small sane default (we test the
  *structural* hypothesis first; tuning comes only if the signal is promising).
