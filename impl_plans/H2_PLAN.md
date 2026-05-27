# H2 Implementation Plan — Physics-Derived Input Channels (ΔQ(V), dQ/dV)

> **Hypothesis (H2):** Adding domain-derived per-cycle channels — the
> incremental-capacity curve `dQ/dV` and the cycle-to-cycle capacity–voltage
> difference `ΔQ(V)` — on top of raw V/I/Q will beat SOTA on **Li-ion**. This is the
> single most established prior in battery-life ML (Severson et al. 2019 predict log
> cycle-life from `Var(ΔQ(V))` with a *linear* model at ~9% error), and the benchmark
> currently omits it. See `../HYPOTHESES.md` for full motivation.
>
> **Feasibility already proven** (subset analysis): `ΔQ₁₀₀₋₁₀(V)` and `dQ/dV` were
> computed cleanly from a real MATR cell's stored `V(t)`/`Q(t)`. The remaining risk is
> *numerical robustness*, not possibility.

## Design decisions (drive the whole plan)

1. **Toggle as a flag, default = current behaviour.** Add
   `--input_channels {base, physics}` (default `base`). `base` = today's `[V, I, Q]`
   (3 channels, bit-for-bit SOTA); `physics` appends derived channels. Zero regression risk.
2. **Feature logic is a pure, testable function**, isolated in `utils/physics_features.py`,
   operating on the assembled `[L, 3, 300]` per-battery array → `[L, C, 300]`. The
   `data_loader.py` edit is then a *single call behind the flag*. This is critical because
   `ΔQ(V)` is **cross-cycle** (needs a reference cycle), so it must run after all cycles of
   a battery are built — a clean second pass, not inside the per-cycle loop.
3. **Not param-matched, and that's expected.** Adding input channels enlarges the first
   embedding layer. Unlike H1, the control here is "same architecture, inputs differ,"
   not "same params." State this explicitly when reporting.
4. **Composes with H1.** By routing both the `linear` (`Linear(L*C)`) and the
   `ConvIntraEncoder` (`num_var=C`) through a dynamic channel count, H2 works on top of
   the H1 conv encoder. We test H2 on the *current* SOTA first, then optionally stacked.

## Channel definitions (to finalize in Phase 0)

Each cycle's 300-pt curve = `[charge 150 | discharge 150]` of `(V, I, Q)`; the loader
already separates and orders charge/discharge (`data_loader.py:562-616`).

- **`dQ/dV`** (incremental capacity, per cycle): differentiate capacity w.r.t. voltage on
  the discharge segment, mapped back onto the 300-pt index. Differentiation amplifies
  noise → apply light smoothing before/after.
- **`ΔQ(V)`** (cross-cycle): this cycle's `Q(V)` minus a **reference early cycle's** `Q(V)`
  (default: first available cycle; Severson uses cycle 10 vs 100), interpolated onto a
  **common monotonic voltage grid**. The reference cycle's own `ΔQ(V)` is 0.

## Hard constraints to respect

- **No GPU locally + DeepSpeed hardwired** → only CPU correctness tests here; full training
  on the CUDA box. **Subset data already downloaded** → the *feature function* (Phase 1) can
  be validated on real pkls **now**, a genuine advantage over H1.
- **Must compare against a reproduced baseline** (`--input_channels base`), never paper numbers.
- Don't break `base` — it stays the default and unchanged.

---

## Phase 0 — Prerequisites & guardrails
*Lock the feature definitions and integration contract before coding.*

- [x] Created git branch `h2-physics-features`.
- [x] Finalized: channels = `dqdv` (per-cycle dQ/dV) + `deltaq` (ΔQ(V) vs reference cycle,
      idx 0); computed **per-segment** (charge `:150`, discharge `150:`); ΔQ aligned in
      **voltage space** (interp ref Q onto this cycle's V), not by index.
- [x] Confirmed build site (`data_loader.py:618-626`) and the hardcoded `*3` in
      `CPMLP.py`/`CPTransformer.py` (`intra_embed`, `ConvIntraEncoder num_var`).
- [x] Normalization decided: new channels left raw + clipped to ±50 (dQ/dV) with NaN/inf
      guards; ΔQ inherits normalized-Q units. **Scaling flagged as a tuning axis** (dQ/dV
      saturated the clip on the real cell — see EXPERIMENT.md threats).
- [x] Noted: H2 is feature-addition (NOT param-matched); stacked H1+H2 run planned (Phase 6).

## Phase 1 — Physics feature module (the core of H2)  ✅ DONE
*One pure, numerically-robust, unit-tested function. Validate on the real subset now.*

- [x] Added `utils/physics_features.py` with `add_physics_channels(curves, channels, ref_cycle_idx)`:
      `[L,3,300]` → `[L,5,300]`. Helpers `_dqdv` (segment-wise dQ/dV) and `_deltaq`
      (voltage-aligned cross-cycle difference).
- [x] Robust numerics: per-segment compute, sorted-V interpolation for ΔQ, flat-V guard
      (|dV|<eps→eps), `np.nan_to_num` + clip on output.
- [x] **Validated on real MATR cell** (smoke test check #2): output `[100,5,300]`, all finite,
      ΔQ range [-0.015, 0.062] (sane normalized-Q units); reference-cycle ΔQ ≈ 0 confirmed.

## Phase 2 — Wire features into the data pipeline  ✅ DONE
*Single call behind the flag; make the channel count dynamic.*

- [x] Added `--input_channels {base, physics}` in `run_main.py` (default `base`); dataset reads
      it via `self.args` (already stored in `Dataset_original.__init__`).
- [x] `data_loader.py:get_charge_discharge_curves` calls `add_physics_channels(curves)` when
      `physics`. Per-cycle zero-padding (`:622`) stays 3-ch; physics runs on the assembled
      `[L,3,300]` and padded cycles are re-zeroed across all channels by the collate's
      `curve_attn_mask` broadcast — so no `(C,len)` padding change needed.
- [x] `enc_in = C` propagated via the launch scripts (`--enc_in 5` for physics).

## Phase 3 — Make the models channel-count-agnostic  ✅ DONE
*So both linear and conv intra-encoders accept C channels (composes with H1).*

- [x] Replaced hardcoded `*3` with `self.enc_in = getattr(configs,'enc_in',3)` in
      `CPMLP.py`/`CPTransformer.py` (`intra_embed = Linear(charge_discharge_length*enc_in, ...)`).
- [x] `ConvIntraEncoder(num_var=self.enc_in, ...)`.
- [x] Confirmed `enc_in=3` reproduces the current models (smoke test builds both at enc_in=3).

## Phase 4 — Run-script & config plumbing  ✅ DONE
- [x] Created `experiments/h2/CPMLP_phys.sh` and `CPTransformer_phys.sh` with
      `--input_channels physics --enc_in 5` and `*_phys` ids.
- [x] Originals (`--input_channels base`, `enc_in=3`) are the A/B baseline.
- [x] 3-seed × 4-domain expressible via `--seed` / `--dataset`.

## Phase 5 — Local CPU test (correctness, not performance)  ✅ DONE
- [x] `experiments/h2/smoke_test.py`: (1) synthetic feature check, (2) **real MATR cell**
      feature check, (3) CP{MLP,Transformer} × enc_in{3,5} × intra{linear,conv} forward+backward.
      All output `[B,1]`, finite loss, grads. **ALL CHECKS PASSED.**
- [x] `enc_in=3 / base` path builds identically to SOTA (verified in check #3).

## Phase 6 — Full training runs (GPU machine)
- [ ] Reproduce baseline (`base`) for CPMLP & CPTransformer per domain.
- [ ] Run the A/B: `{base, physics} × {CPMLP, CPTransformer} × {4 domains} × {3 seeds}`.
- [ ] Optional stacked run: `physics + conv` (H1+H2) to test compounding.
- [ ] Log MAPE + 15%-Acc, mean ± std.

## Phase 7 — Analysis & go/no-go decision
- [ ] Results table: physics vs base, per domain, both backbones.
- [ ] **Falsification (from HYPOTHESES.md):** H2 fails on Li-ion if physics does not improve
      MAPE beyond noise. Separately report whether features help/hurt Zn-ion / Na-ion / CALB.
- [ ] Record outcome in `../HYPOTHESES.md` and memory; if confirmed, fold channels into the
      shared input used by H3.

---

## Deliverables (as built)
- `utils/physics_features.py` — `add_physics_channels` (+ subset validation)  ✅
- edits to `data_provider/data_loader.py`, `run_main.py`, `models/CPMLP.py`, `models/CPTransformer.py`  ✅
- `experiments/h2/CPMLP_phys.sh`, `experiments/h2/CPTransformer_phys.sh`  ✅
- `experiments/h2/smoke_test.py`  ✅
- `experiments/h2/EXPERIMENT.md` — write-up  ✅
- results table + decision logged into `../HYPOTHESES.md` — pending GPU

## Out of scope (explicitly)
- H1 (encoder), H3 (multi-domain), H4 (loss) — separate plans, though H1+H2 stacking is a
  planned optional run.
- Exhaustive feature engineering (e.g. temperature/IR-derived channels): test the canonical
  `dQ/dV` + `ΔQ(V)` first; expand only if the signal is promising.
