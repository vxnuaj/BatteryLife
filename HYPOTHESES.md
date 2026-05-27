# Hypotheses for Beating the BatteryLife Benchmark

> Goal: beat the state-of-the-art on the **BatteryLife** (KDD 2025) battery life
> prediction benchmark. This document records our top 3 hypotheses. Each is written
> to stand on its own — read any one in isolation.

---

## Shared context (read once)

**The task.** Given the first `S ≤ 100` cycles of a battery's degradation test,
predict its life `y` (number of cycles until it reaches 80% state-of-health; 90%
for the CALB domain). Each cycle is resampled to **300 points** (150 charge + 150
discharge) of **3 variables**: voltage, current, capacity. So one battery is a
tensor `[100 cycles, 3 vars, 300 points]`.

**The current SOTA ("CyclePatch").** Treat each cycle as one token. An
*intra-cycle encoder* turns each cycle's `3×300 = 900` numbers into one embedding;
an *inter-cycle encoder* (MLP / Transformer / RNN) then models the sequence of ~100
cycle-embeddings; a linear layer projects to the predicted life. The two winning
variants are **CPMLP** and **CPTransformer**.

**Targets to beat** (MAPE ↓ / 15%-Acc ↑, per domain):

| Domain  | Best MAPE | Best 15%-Acc | Held by        | Cells |
|---------|-----------|--------------|----------------|-------|
| Li-ion  | 0.179     | 0.620        | CPMLP          | 837   |
| Zn-ion  | 0.515     | 0.297        | CPTransformer / CPMLP | 95 |
| Na-ion  | 0.255     | 0.406        | CPTransformer  | 31    |
| CALB    | 0.140     | 0.704        | CPMLP          | 27    |

**Key facts the paper established** (these constrain what is worth trying):
- Life information lives in **fine-grained interactions among V/I/Q within a
  cycle** — *not* in trend/seasonal structure. Decomposition-based models
  (DLinear, Autoformer, MICN) are therefore weak.
- **RevIN / instance normalization is harmful** — it destroys cross-variable
  interactions. Do not reintroduce it.
- Removing *either* the intra- or inter-cycle encoder hurts; both matter.
- **No single model wins all four domains** — CPMLP and CPTransformer split them.

**Experimental protocol (must hold for every hypothesis below).** Use the
benchmark's exact setup: 6:2:2 random split, Adam, **3 random seeds**, report
**mean ± std** on the test set. Always compare against a *reproduced* CPMLP /
CPTransformer baseline run inside our own harness, never against paper numbers
alone.

---

## Hypothesis 1 — A structure-aware intra-cycle encoder

**Statement.**
Replacing the intra-cycle encoder's `flatten → Linear` step with a structure-aware
encoder (a small 1-D CNN, or a tiny per-cycle attention, over the `[3, 300]`
cycle signal) will lower MAPE across domains. Because the intra-cycle encoder is
**shared by the entire CyclePatch family**, this single change should lift CPMLP
and CPTransformer simultaneously.

**Why it should work (evidence).**
The paper argues life information is carried by *fine-grained, time-aligned
interactions among voltage, current, and capacity*. Yet the current intra-cycle
encoder flattens the `3 × 300` matrix into a length-900 vector and applies one
dense `Linear`. Flattening **discards the alignment** between V, I, and Q at the
same time step — exactly the signal the authors say matters. A convolution over
the 300 points with 3 input channels keeps each variable's local temporal shape
*and* mixes the three variables at each position, giving the network the right
inductive bias instead of forcing it to recover structure from a scrambled vector.

**What to build.**
Keep the CyclePatch scaffold (segment → intra-encoder → inter-encoder →
projection) unchanged. Swap only the per-cycle embedding: feed each cycle as a
`[batch, 3, 300]` tensor into a few `Conv1d` layers (e.g. channels `3 → 64 → 128`,
small kernels, with pooling) and flatten to the model dimension `D`. Reuse the
existing residual feed-forward stack on top if helpful. The inter-cycle encoder
(MLP for CPMLP, Transformer for CPTransformer) stays identical.

**Prediction.**
Broad MAPE reduction; largest where good per-cycle features matter most —
data-rich **Li-ion** and data-scarce **Zn-ion**. Expected to improve *both*
CPMLP and CPTransformer.

**How to falsify.**
Train CP{MLP,Transformer} with the new intra-encoder vs. the reproduced baseline,
3 seeds, all four domains. Falsified if MAPE does not improve (within noise) on at
least Li-ion and one small domain.

**Risk / caveats.** Low. A dense linear *can* in principle approximate the
interaction, so gains on Li-ion (abundant data) may be modest; the test is whether
the inductive bias helps most on the small domains. Watch parameter count — keep
the conv encoder small so the comparison is about structure, not capacity.

---

## Hypothesis 2 — Physics-derived input channels (ΔQ(V) and dQ/dV)

**Statement.**
Adding domain-derived channels — the incremental-capacity curve `dQ/dV` and the
cycle-to-cycle capacity-voltage difference `ΔQ(V)` — to each cycle token, on top
of raw V/I/Q, will beat SOTA on **Li-ion**.

**Why it should work (evidence).**
The benchmark's own seminal reference (Severson et al., 2019) predicts *log cycle
life* from the **variance of `ΔQ(V)` between an early and a later cycle**, using
only a **linear** model, at ~9% error. That is a remarkably strong, well-validated
signal. The BatteryLife models receive only raw V/I/Q and must rediscover this
relationship from scratch. Supplying `ΔQ(V)` and `dQ/dV` directly injects the
single most established prior in battery-life ML — a cheap, high-value inductive
bias the benchmark currently omits.

**What to build.**
For each cycle, compute two extra per-point channels from the already-resampled
curves: (1) `dQ/dV` (incremental capacity, the discharge capacity differentiated
with respect to voltage), and (2) `ΔQ(V)` = this cycle's `Q(V)` minus a reference
early cycle's `Q(V)` (e.g. cycle 1 or 10), interpolated onto a common voltage
grid. Concatenate as additional input variables so each cycle token becomes
`[5, 300]` (or more) instead of `[3, 300]`. No architecture change required —
only the input tensor and the first embedding's input width.

**Prediction.**
Substantial MAPE gain on **Li-ion** (the chemistry these features were designed
for). Effect on Zn-ion / Na-ion is genuinely unknown (different electrochemistry)
and is itself worth measuring.

**How to falsify.**
Train the current SOTA model with vs. without the extra channels, 3 seeds.
Falsified on Li-ion if MAPE does not improve beyond noise. Separately report
whether the features help or hurt the non-Li-ion domains.

**Risk / caveats.** Medium. The features are chemistry-specific and require careful,
numerically-stable computation from resampled curves (differentiation amplifies
noise; `ΔQ(V)` needs a clean voltage-grid interpolation). They may not transfer to
Zn/Na chemistries. High value-of-information regardless of outcome.

---

## Hypothesis 3 — Joint multi-domain training with domain conditioning

**Statement.**
A single model trained **jointly on all four domains** — with a shared intra-cycle
encoder and a domain-conditioned inter-cycle path (a learned domain embedding /
FiLM modulation, or per-domain output heads) — will beat the per-domain SOTA on
the data-scarce **Zn-ion** and **Na-ion** domains, without degrading Li-ion.

**Why it should work (evidence).**
Zn-ion (95 cells across 45 chemical systems ≈ 2 cells/system) and Na-ion (31 cells)
are the weakest domains purely because of **data scarcity**. The within-cycle
electrochemical signal (the shape of V/I/Q over a cycle) is plausibly *shared*
across chemistries, so a low-level encoder trained on the 837 Li-ion cells should
benefit the small domains. Crucially, the paper only tried **sequential** transfer
(freeze / fine-tune / domain-adaptation from Li-ion → target), and reported it
**fails** — but it never tried **joint** training, where domain conditioning
absorbs the distribution shift while the shared encoder pools statistical strength.

**What to build.**
One model, trained on the union of all four domains. Add a domain identifier per
sample → a learned domain embedding used to modulate the inter-cycle encoder
(FiLM-style scale/shift, or simply concatenated), and/or a separate small
projection head per domain. The intra-cycle encoder is fully shared. Balance the
sampler so small domains are not swamped by Li-ion. Evaluate per-domain against
each domain's own SOTA.

**Prediction.**
Largest gains on **Zn-ion** and **Na-ion** (most headroom); Li-ion held roughly
constant. A win here is also the cleanest *narrative* SOTA claim — one robust model
beating four specialised ones.

**How to falsify.**
Compare the jointly-trained, domain-conditioned model per-domain against
reproduced single-domain SOTA, 3 seeds. Falsified if the small domains do not
improve, or if Li-ion regresses materially (i.e. negative transfer dominates).

**Risk / caveats.** Medium-high. Negative transfer is a real possibility given how
different the chemistries are; the domain conditioning and sampler balance are the
levers that decide success. Directly targets the paper's stated open problem of
handling multiple domains at once.

---

## Notes

- A cheap, broadly-applicable companion knob — **log-target training** (MSE on
  standardized `log y` instead of `y`, to match the heavy-tailed label
  distribution and the relative MAPE metric) — is worth toggling alongside any of
  the above. The repo already exposes `MAPE` and `BMSE` loss options, so verify
  what has already been tried before investing.
- **H1 and H2 compose** (better encoder + better inputs) and together are the most
  likely route to a clean multi-domain SOTA. **H3** is the highest-upside play on
  the weak domains and the strongest stand-alone research story.
- Prerequisite for all empirical work: download the processed dataset
  (HuggingFace `Hongwxx/BatteryLife_processed` or Zenodo), secure a GPU, and
  **reproduce the CPMLP / CPTransformer baselines first**.
