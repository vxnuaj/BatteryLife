# GPU Runbook — running the experiments end to end

Concrete operator checklist for running the H1/H2/H3 experiments on a GPU box. For *what*
the experiments are, see [`README.md`](README.md) and each `hN/EXPERIMENT.md`; for project
context see [`../CLAUDE.md`](../CLAUDE.md). This file is the *how to actually run it*.

> Status reminder: H1 and H2 run as-is. **H3's training-loop glue was drafted but never
> executed** — validate it on a subset before trusting it (step 4).

## 1. Environment
```bash
git clone https://github.com/vxnuaj/BatteryLife.git
cd BatteryLife && git checkout all-experiments
pip install -r requirements.txt          # torch, accelerate, deepspeed, peft, wandb, ...
pip install BatteryML                     # per the repo README (preprocessing dep, imported by data_loader)
```
The CPU smoke tests only need `torch numpy` (+`reformer_pytorch`); full training needs the
above plus `sklearn`, `scipy`, `matplotlib`, `denseweight` (all in requirements).

## 2. accelerate / DeepSpeed for YOUR GPU count
Training launches via `accelerate launch --multi_gpu` with DeepSpeed ZeRO-2
(`ds_config_zero2_baseline.json`, hardcoded in `run_main.py`). The scripts assume **2 GPUs**.
You must match them to your hardware (next step). If you hit DeepSpeed/accelerate issues,
`accelerate config` once, or run single-GPU by setting the script vars below.

## 3. EDIT THE LAUNCH SCRIPTS (required — they ship with placeholders)
Each script in `train_eval_scripts/*.sh` and `experiments/h{1,2,3}/*.sh` has:

| Line | Placeholder | Fix |
|---|---|---|
| `checkpoints=...` | `/path/to/your/saving/folder` (CPTransformer.sh: stale `/data/hwx/BL_new`) | a real writable path |
| `num_process=2` | assumes 2 GPUs | your GPU count (e.g. `1`) |
| `CUDA_VISIBLE_DEVICES=0,1` | (CPTransformer.sh: `2,3`) | the GPU ids you actually have (e.g. `0`) |

For a single GPU: `num_process=1`, `CUDA_VISIBLE_DEVICES=0` (you may also drop `--multi_gpu`).

## 4. Data
```bash
hf auth login                             # + accept terms at huggingface.co/datasets/Battery-Life/BatteryLife_Processed
# Skip the 24.3 GB the benchmark never uses, then slim + cache (see DATASET_OPTIMIZATION.md):
hf download Battery-Life/BatteryLife_Processed --repo-type dataset --local-dir dataset/ \
    --exclude "SDU/*" "Stanford_2/*"
python scripts/build_slim_dataset.py --root dataset/        # truncate to 100 cycles (lossless)
# then add --use_cache to runs; the first build lets you run from a few hundred MB.
```
`dataset/` is gitignored except the tracked `seen_unseen_labels/` (needed by eval).
**Full storage/speed plan + numbers: [`DATASET_OPTIMIZATION.md`](DATASET_OPTIMIZATION.md)**
(86 GB → a few hundred MB, identical results). The naive full download still works if you prefer.

## 5. Run order (do these in sequence)

1. **CPU sanity** — `python experiments/h{1,2,3}/smoke_test.py` → all `ALL CHECKS PASSED`.
2. **H3 glue validation** — run `experiments/h3/CPMLP_multidomain.sh` on a **small data subset / few
   epochs first**. Expect to debug the never-run integration (per-domain scaler, `my_collate_fn_multidomain`,
   FiLM path, per-domain metric logging). Only proceed once a few iterations run + log per-domain metrics.
3. **Reproduce the baselines** — run the un-flagged single-domain models
   (`train_eval_scripts/CPMLP.sh`, `CPTransformer.sh`) per domain/seed and confirm they land near the
   paper SOTA (Li-ion 0.179 / Zn 0.515 / Na 0.255 / CALB 0.140 MAPE). **We compare against these, not the paper.**
4. **Run the treatments + ablations**, each over **3 seeds (42/2021/2024) × 4 domains**:
   - H1: `experiments/h1/*_conv.sh`
   - H2: `experiments/h2/*_phys.sh`  (+ planned random-noise-channel control)
   - H3: `experiments/h3/*_multidomain.sh` (dataset `MIX_all` / `MIX_all42` / `MIX_all2024` per seed)
     + ablations: w/o conditioning (`--multidomain off` on `MIX_all`), w/o balanced sampling,
       `--label_scaling global`
   - Stacked: combine flags (`--intra_encoder conv --input_channels physics --multidomain on`)
5. **Record** mean ± std into each `hN/EXPERIMENT.md` results table; decide go/no-go against that
   file's falsification criteria.

## 6. Known caveats to watch (likely to need iteration)
- **H2 `dQ/dV` scaling:** saturated its ±50 clip on real data → if H2 underperforms, try
  per-channel normalization of `dQ/dV` before concluding it failed (see `h2/EXPERIMENT.md`).
- **Hyperparameters** were tuned for the baseline; a variant may need a light sweep before being
  declared a failure (don't over-tune — see each EXPERIMENT.md's "garden of forking paths" note).
- **H2 + augmentation:** the `cj_aug` curve path is still 3-channel; baseline runs don't use it,
  but enabling augmentation with `--input_channels physics` would need matching channel handling.
- **H3 per-domain scaling** (`--label_scaling per_domain`) and the `args._domain_means/_stds` stash
  are new code — verify val/test reuse TRAIN stats correctly during the step-2 validation run.
