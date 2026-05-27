# H3 Implementation Plan — Joint Multi-Domain Training with Domain Conditioning

> **Hypothesis (H3):** A single model trained **jointly on all four domains** — with a
> shared intra-cycle encoder and a **domain-conditioned** inter-cycle path — will beat the
> per-domain SOTA on the data-scarce **Zn-ion** and **Na-ion** domains, *without* degrading
> Li-ion. The within-cycle electrochemical structure is partially shared across chemistries
> (so a shared encoder pools strength from the 837 Li-ion cells), while domain conditioning
> supplies the minimal freedom to absorb the between-domain distribution shift. The paper
> only tried **sequential** transfer (frozen / fine-tune / domain-adaptation), which fails;
> **joint** training was never tried. See `../HYPOTHESES.md`.

## Why H3 is harder than H1/H2 (and what that implies)

H1 was architecture-only; H2 added a data-pipeline feature pass. H3 is **cross-cutting** —
it touches data (a new combined split, a domain id, per-domain label scaling), the model
(a new `domain_id` forward argument + conditioning), the training loop (pass the id,
balance the sampler), and evaluation (break metrics out per domain). The plan front-loads
the design decisions because they interact.

## Design decisions (drive the whole plan)

1. **Default off, single-domain unaffected.** A flag `--multidomain {off, on}` (default
   `off`). When `off`, training is bit-for-bit today's per-domain behaviour. Conditioning
   modules are only built/active when `on`.
2. **Conditioning mechanism = FiLM from a domain embedding.** Learn a small embedding per
   domain (4 domains) → produce per-domain scale/shift (FiLM) applied to the inter-cycle
   features. Rationale (hard to vary): FiLM is the *minimal* mechanism that lets one shared
   network specialise per domain without separate weight sets; it is the part whose removal
   the experiment will test. (Alternatives to ablate: concat-embedding, per-domain heads.)
3. **Label scaling = per-domain (primary).** Fit one `StandardScaler` **per domain**, not
   one global scaler — otherwise the small-life domains (median ~170) are squashed against
   Li-ion (up to ~5000). Predictions are inverse-transformed with the sample's own domain
   scaler. (Alternative to compare: global log-target, ties to H4.)
4. **Balanced sampling by domain.** Li-ion (837) would swamp Na-ion (31). Reuse the existing
   `--weighted_sampling` machinery to upweight scarce domains so each contributes
   comparably per epoch.
5. **Per-domain evaluation.** Metrics (MAPE, 15%-Acc) must be computed **separately per
   domain** on the joint test set, to compare against each domain's SOTA. Reuse the existing
   `seen_unseen`-style gather to also gather `domain_id`.
6. **Shared intra-cycle encoder.** The intra-cycle encoder (linear or H1-conv) is fully
   shared across domains — this is the pooling mechanism and composes with H1/H2.

## Hard constraints

- No GPU locally + DeepSpeed hardwired → CPU correctness only; full training on the CUDA box.
- Subset data only → the joint **loader** can't be fully exercised locally; the conditioning
  model, the per-domain metric function, and the split-construction logic *can* be unit-tested.
- Compare against **reproduced** per-domain SOTA, never paper numbers.

---

## Phase 0 — Guardrails & design lock  ✅ DONE
- [x] Branched `h3-multidomain`.
- [x] Domain map locked in `utils/domain_utils.py:filename_to_domain_id` (0=Li-ion, 1=Zn, 2=Na, 3=CALB).
- [x] Seed handling: `MIX_all` (2021) / `MIX_all_42` / `MIX_all_2024` already exist in
      `data_split_recorder.py` = `MIX_large` + seed-matched Zn/CALB/Na.
- [x] Integration points confirmed. **Discovered:** training uses `my_collate_fn_baseline`
      (7-tuple, **no** `dataset_id`) and the train loop unpacks exactly that — so threading a
      domain id changes shared, GPU-only training code (see Phases 3/4 deferral).

## Phase 1 — Joint dataset: combined split, domain id, per-domain scaling  ✅ DRAFTED
- [x] Combined split present; added `MIX_all` / `MIX_all42` / `MIX_all2024` branches in
      `data_loader.py` (verified: `MIX_all` = 612 train cells, all 4 domains).
- [x] `domain_id` mapping implemented + unit-tested (`filename_to_domain_id`).
- [x] `domain_id` built per-sample in `read_data`, exposed in `__getitem__`, carried by a
      **new** `my_collate_fn_multidomain` (8-tuple) — baseline collate untouched.
- [x] Per-domain label scalers in `data_loader.py` (`--label_scaling per_domain`): train fits
      per-domain mean/std; val/test reuse train stats stashed on `args`. *Drafted, compile-clean,
      needs a GPU smoke-run.*

## Phase 2 — Domain-conditioning module + model wiring  ✅ DONE
- [x] `layers/domain_conditioning.py` — `FiLMConditioner` (zero-init = identity).
- [x] Added `--multidomain {off,on}`, `--domain_cond {film,concat,heads}`, `--num_domains`,
      `--label_scaling {global,per_domain,log}` in `run_main.py`.
- [x] Wired into `CPMLP.py`/`CPTransformer.py`: `forward(..., domain_id=None)`; FiLM applied to
      the cycle-token embeddings when `multidomain on` and `domain_id` given. Default unchanged.

## Phase 3 — Training loop: pass domain id + balanced sampling  ✅ DRAFTED
- [x] `run_main.py` train loop unpacks `domain_id` (guarded; 7-tuple default) and passes it to
      `model(...)` only when present (non-CP models never receive the kwarg).
- [x] Domain-balanced sampling via `WeightedRandomSampler` (inverse domain frequency) in
      `data_factory`, active only under `--multidomain on --weighted_sampling`.
- [x] Per-domain inverse-transform in the train-loop metric logging.
> All guarded so the default path is byte-identical. **Drafted, compile-clean, not yet run on GPU.**

## Phase 4 — Per-domain evaluation  ✅ DRAFTED
- [x] `per_domain_metrics(preds, targets, domain_ids)` implemented + unit-tested (`domain_utils.py`).
- [x] Wired into `vali_baseline` (`utils/tools.py`): guarded 8-tuple unpack, per-domain
      inverse-transform, gathers `domain_id`, **logs** per-domain MAPE/Acc (return signature
      unchanged so `run_main` is unaffected). *Drafted, needs GPU smoke-run.*

## Phase 5 — Run scripts & config  ✅ DONE
- [x] `experiments/h3/CPMLP_multidomain.sh`, `CPTransformer_multidomain.sh`
      (`--dataset MIX_all --multidomain on --num_domains 4 --label_scaling per_domain --weighted_sampling`).
- [x] Per-domain SOTA baselines = existing single-domain scripts.
- [x] 3-seed sweep expressible via `MIX_all` / `MIX_all42` / `MIX_all2024` + `--seed`.

## Phase 6 — Local CPU test (correctness, not performance)  ✅ DONE
- [x] `experiments/h3/smoke_test.py`: FiLM (identity@init + domain-sensitive), utils, combined-split
      counts/coverage, and CP{MLP,Transformer} on-path forward+backward + domain-sensitivity.
      **ALL CHECKS PASSED.** H1/H2 smoke tests still pass (off-path preserved).
- [x] Confirmed `multidomain off` / `domain_id=None` reproduces single-domain models.

## Phase 7 — Full training runs (GPU machine)
- [ ] Reproduce per-domain SOTA baselines (already covered by H1/H2 baseline runs).
- [ ] Joint runs: `{CPMLP, CPTransformer} × {4 domains evaluated} × {3 seeds}`, `multidomain on`.
- [ ] Ablations (the hard-to-vary tests): **w/o conditioning** (shared encoder, no FiLM),
      **w/o shared encoder** (per-domain encoders ≈ separate models), **global vs per-domain
      scaler**, **w/o balanced sampling**.
- [ ] Optional: stack with H1 conv encoder and/or H2 physics channels.

## Phase 8 — Analysis & go/no-go decision
- [ ] Per-domain table: joint vs per-domain SOTA, both backbones, mean ± std.
- [ ] **Falsification (from HYPOTHESES.md):** H3 fails if the scarce domains (Zn-ion / Na-ion)
      do **not** improve, **or** if Li-ion regresses materially (negative transfer dominates).
- [ ] Mechanism checks: removing conditioning should hurt (else FiLM isn't doing the work);
      removing sharing should erase the gain (else pooling isn't the mechanism).
- [ ] Record outcome in `../HYPOTHESES.md` and memory.

---

## Deliverables
- `layers/domain_conditioning.py` — `FiLMConditioner`
- edits to `data_provider/data_split_recorder.py`, `data_provider/data_loader.py`,
  `run_main.py`, `utils/tools.py`, `models/CPMLP.py`, `models/CPTransformer.py`
- `experiments/h3/CPMLP_multidomain.sh`, `CPTransformer_multidomain.sh`
- `experiments/h3/smoke_test.py`
- `experiments/h3/EXPERIMENT.md` — write-up (hard-to-vary hypothesis, ablations, results)
- results + decision logged into `../HYPOTHESES.md`

## Out of scope (explicitly)
- H1 / H2 / H4 as standalone experiments — though H3 is designed to **compose** with the H1
  conv encoder and H2 physics channels (all gated by their own flags).
- Heavier domain-generalization methods (adversarial alignment, meta-learning) — start with
  the minimal FiLM-conditioning + joint-training claim; escalate only if it shows signal.
- The seen/unseen *aging-condition* split (a finer-grained generalization axis) — orthogonal;
  could be a follow-up once domain-level joint training is understood.
