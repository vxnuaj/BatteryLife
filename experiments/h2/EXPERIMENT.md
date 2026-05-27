# Experiment H2 — Physics-Derived Input Channels (dQ/dV, ΔQ(V))

| | |
|---|---|
| **Status** | Implemented & CPU-verified (Phases 0–5). Training (Phases 6–7) pending GPU + full data. |
| **Branch** | `h2-physics-features` |
| **Created** | 2026-05-27 |
| **Plan** | [`../../impl_plans/H2_PLAN.md`](../../impl_plans/H2_PLAN.md) |
| **Parent doc** | [`../../HYPOTHESES.md`](../../HYPOTHESES.md) |
| **Benchmark** | BatteryLife (KDD 2025), arXiv 2502.18807 |

---

## 1. Problem context

BatteryLife feeds models only the **raw** resampled signals per cycle — voltage,
current, capacity `[V, I, Q]` — and asks them to predict cycle-life. The benchmark's
own seminal reference (Severson et al., 2019) showed that a **linear** model on a single
hand-built feature — the **variance of ΔQ(V)**, the cycle-to-cycle change in the
capacity–voltage curve — predicts *log* cycle-life at ~9% error. The closely related
**dQ/dV** incremental-capacity curve is the standard diagnostic whose peaks correspond to
electrode phase transitions and whose shift/shrinkage tracks degradation.

Neither is provided to the BatteryLife models. They must rediscover these transforms from
raw `[V, I, Q]`.

## 2. Hypothesis (hard to vary)

> **Claim.** Appending two derived channels — per-cycle `dQ/dV` and cross-cycle `ΔQ(V)` —
> to the raw `[V, I, Q]` input improves cycle-life prediction on **Li-ion**, because these
> are the *specific nonlinear transforms of (V, Q)* that the established electrochemical
> mechanism makes life-predictive, and which a network cannot cheaply synthesize from raw
> inputs when data is limited.

Decomposed into load-bearing parts:

1. **Degradation manifests as a shift of the Q(V) curve across cycles** (loss of lithium
   inventory / active material shrinks and translates the capacity–voltage relationship).
   `ΔQ(V)` measures exactly that shift; its variance is a validated life predictor.
2. **`dQ/dV` exposes phase-transition peaks** whose movement encodes the *mechanism* of
   degradation, not just its magnitude.
3. **Both are nonlinear, alignment-dependent functions of the raw inputs.** `ΔQ(V)` requires
   (a) aligning two cycles **in voltage space** and (b) subtracting; `dQ/dV` requires
   differentiation. A model given only raw `[V,I,Q]` must *learn* to align-in-voltage and
   differentiate — feasible with 837 Li-ion cells, hard with 31–95 Zn/Na cells.

### Why this is hard to vary (Deutsch criterion)

Each part is tied to a distinct, breakable prediction:

- The mechanism is specifically about **voltage-space alignment**. So computing ΔQ by
  **index** instead of by voltage (`Q_cycle[i] − Q_ref[i]`) should help **much less** — even
  though it is also a "cross-cycle difference." If index-ΔQ matched voltage-ΔQ, part 1's
  *mechanism* (not just the existence of a difference signal) would be false. → ablation P3.
- The Severson result singles out **ΔQ(V)** specifically. So ΔQ(V) should carry **more** of
  the gain than dQ/dV. If dQ/dV alone captured everything and ΔQ(V) added nothing, the
  appeal to the Severson mechanism would be undercut. → ablation P2.
- The features are **chemistry-specific** (dQ/dV peaks and Q(V) shapes differ for Zn/Na). So
  the gain is predicted for **Li-ion** and is **uncertain or absent** for Zn/Na. A naive
  "more features help everything" would be easy to vary and would wrongly predict uniform
  gains. Our claim forbids that and treats a Zn/Na non-gain as *consistent*, not refuting.

## 3. Risky predictions

- **P1 (primary).** Physics channels lower **Li-ion** MAPE vs. the `base` (3-channel) model,
  for both CPMLP and CPTransformer.
- **P2 (ablation).** `ΔQ(V)`-only ≳ `dQ/dV`-only in Li-ion gain (Severson's feature dominates).
- **P3 (mechanism-breaking ablation).** **Voltage-aligned** ΔQ(V) > **index-aligned** ΔQ.
  If equal, the "voltage-space" mechanism is wrong.
- **P4 (control).** Replacing the 2 physics channels with 2 **random noise** channels yields
  **no** gain over base — isolating that any improvement comes from *information*, not merely
  from a wider input layer (addresses the non-param-matched confound, §4).
- **P5.** Gain may not transfer to Zn-ion/Na-ion; if physics *hurts* there, that is consistent
  with chemistry-specificity (part 3), not a refutation of P1.

## 4. Method

**Controlled variable.** A single flag `--input_channels {base, physics}` (default `base`).
`base` is bit-for-bit the current SOTA input (`[V,I,Q]`, `enc_in=3`); `physics` appends
`dQ/dV` and `ΔQ(V)` (`enc_in=5`). Everything else is identical.

**Confound — not param-matched (acknowledged).** Adding channels enlarges the first
embedding layer, so the physics model has *more* parameters than base. Unlike H1 (where we
shrank the encoder to control capacity), here the control is the **noise-channel baseline
(P4)**: if 2 random channels don't help, the gain is from feature *information*, not width.

**Feature definitions** (`utils/physics_features.py`):
- Curve layout: each cycle's 300 points = `[charge 150 | discharge 150]`, channels `[V, I, Q]`.
- `dQ/dV`: computed **per segment**, `grad(Q)/grad(V)` with a flat-voltage guard, clipped to ±50.
- `ΔQ(V)`: per segment, this cycle's `Q` minus the **reference cycle (idx 0)** capacity sampled
  at this cycle's **voltages** (`np.interp` of ref `Q(V)` onto current `V`) — i.e. aligned in
  voltage space. Reference cycle's ΔQ ≈ 0 by construction (verified).

**Architecture.** Models made channel-agnostic (`enc_in` replaces hardcoded `×3`), so physics
works with **both** intra-encoders — `linear` (current SOTA) and `conv` (H1). This enables a
stacked **H1+H2** run.

**Datasets / metrics / protocol.** 4 domains; MAPE (↓ primary) + 15%-Acc (↑); benchmark-exact
6:2:2 split, Adam, 3 seeds (42/2021/2024), mean ± std; compare against reproduced `base`.

## 5. Falsification criteria

- **H2 refuted** if physics does not improve **Li-ion** MAPE beyond noise (P1 fails).
- **Mechanism undermined** (even if P1 holds) if index-ΔQ ≈ voltage-ΔQ (P3), or if the
  **noise-channel control helps as much as physics** (P4 — would mean it was just capacity/width).

## 6. File map

### Experiment artifacts (`experiments/h2/`)
| File | Role |
|---|---|
| `EXPERIMENT.md` | This write-up. |
| `smoke_test.py` | CPU harness: synthetic + **real MATR** feature checks, and CP{MLP,Transformer} × enc_in{3,5} × intra{linear,conv} forward/backward. **Status: PASSED.** |
| `CPMLP_phys.sh`, `CPTransformer_phys.sh` | Launch with `--input_channels physics --enc_in 5`. |

### In-place source changes (shared files — cannot be relocated)
| File | Change |
|---|---|
| `utils/physics_features.py` | **New.** `add_physics_channels` — the core H2 feature code. |
| `run_main.py` | Added `--input_channels {base, physics}` (default `base`). |
| `data_provider/data_loader.py` | `get_charge_discharge_curves` calls `add_physics_channels` when `physics` (single line behind the flag). |
| `models/CPMLP.py`, `models/CPTransformer.py` | `enc_in` replaces hardcoded `×3` for both linear and conv intra-encoders. |

### Baselines (unmodified, `train_eval_scripts/`)
`CPMLP.sh`, `CPTransformer.sh` (run with default `base`, `enc_in=3`).

### Not yet built (planned ablations from §3)
`ΔQ`-only / `dQ/dV`-only, **index-aligned ΔQ** (P3), and **noise-channel** (P4) variants —
small additions to `physics_features.py` + flags; add in the GPU phase if P1 looks promising.

## 7. Reproduction

**Now (CPU) — correctness:**
```bash
python experiments/h2/smoke_test.py        # expect: ALL CHECKS PASSED
```

**Later (GPU + full dataset) — the experiment:**
```bash
sh train_eval_scripts/CPMLP.sh             # base arm
sh experiments/h2/CPMLP_phys.sh            # physics arm
# repeat for CPTransformer; sweep 3 seeds × 4 domains; record MAPE / 15%-Acc.
# stacked H1+H2: add `--intra_encoder conv` to the *_phys scripts.
```

## 8. Results

*Pending GPU runs. Fill mean ± std over 3 seeds; bold the better of {base, physics} per cell.*

| Backbone | Channels | Li-ion MAPE | Li-ion 15%-Acc | Zn-ion MAPE | Na-ion MAPE | CALB MAPE |
|---|---|---|---|---|---|---|
| CPMLP | base (3) | — | — | — | — | — |
| CPMLP | physics (5) | — | — | — | — | — |
| CPTransformer | base (3) | — | — | — | — | — |
| CPTransformer | physics (5) | — | — | — | — | — |

**Reference SOTA (paper):** Li-ion 0.179 / Zn-ion 0.515 / Na-ion 0.255 / CALB 0.140 (MAPE).

## 9. Threats to validity

- **Feature scaling.** On the real MATR cell the `dQ/dV` channel **saturated the ±50 clip**
  (flat-voltage regions). The full loader's per-segment 150-pt resampling should reduce this,
  but **per-channel normalization of dQ/dV is a likely-needed tuning step** — if physics
  underperforms, try standardizing dQ/dV before declaring P1 failed.
- **Reference-cycle choice.** We use cycle idx 0; Severson uses cycle 10 vs 100. With variable
  input length S, idx 0 is always available but may be a formation-ish cycle for some cells.
- **Not param-matched** — mitigated by the P4 noise-channel control, which must be run before
  attributing any gain to the physics information.
- **Augmentation path** (`cj_aug` curves) is left 3-channel; baseline runs don't use it, but
  enabling augmentation with `physics` would need the same channel treatment.
- Small-domain variance (Zn/Na) is large — P2/P3/P5 may need >3 seeds to resolve.

## 10. Changelog
- 2026-05-27 — Phases 0–5 implemented; smoke test passing (incl. real-data feature validation); write-up created.
