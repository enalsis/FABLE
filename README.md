# FABLE: reproducible GPU research package

This package integrates the existing E5 higher-dimensional verifier with an expanded, memorization-grounded E6/E7 benchmark. It has **actual local smoke runs** and a separately configurable **full GPU research protocol**. The latter has **not been executed here**.

## 1. Installation

Requirements: Python 3.10+, PyTorch with your installed CUDA driver, torchvision matched to PyTorch, NumPy, SciPy (HiGHS MILP), scikit-learn, matplotlib. On Windows run the Python commands manually or use WSL2. GPU is for model training; the HiGHS solver runs on CPU. For the precise installation of GPU torch, use https://pytorch.org/get-started/locally/ for your system.

```bash
python -m venv .venv
source .venv/bin/activate
# Install an appropriate CUDA-enabled torch + torchvision first.
pip install -r requirements.txt
python -m unittest discover -s tests -v
bash run_smoke.sh
```

## 2. Full GPU validation

```bash
bash run_gpu_all.sh
```

This runs two independently labeled benchmarks:

- **A. E5 scalability** (`runs/gpu_scalability`): MNIST digits 3/8, PCA dimensions 2/8/16/32, targeted/random-deletion controls, 10 seeds, three retraining references, class-matched loss-AUC diagnostics, per-region coverage, global MILP verification, PGD and random search.
- **B. E7 deletion/memorization** (`runs/gpu_memorization`): 5 deletion requests × 10 model seeds, a *planted and confirmed* incorrect-label trigger, retain-only short/long fine-tuning, head-reset, sham deletion (retains canaries), three retraining references, exactly verified trigger/no-trigger regions using an invertible two-piece nonlinear coupling flow, nearest-canary distances, 5,000-sample/32-restart PGD baselines, and an optional balanced shadow-model audit (16 shadows per seed).

**Resource warning:** The combined preset trains thousands of networks, including 800 shadow networks for the five-request, ten-seed study. Start with a smaller run and scale gradually. Check storage and CPU solver usage. Shadow training is intentionally expensive. The runner selects nonoverlapping canary sets across requests, but shared retained-data training means request outcomes may still be statistically dependent. If needed, skip privacy shadows temporarily with `python src/run_full.py --preset gpu --output runs/gpu_memorization --no-shadow` and run them later without `--no-shadow`.

For a smaller initial GPU run:

```bash
python src/run_full.py --preset gpu --output runs/gpu_pilot --requests 2026 2027 --seeds 11 23 --no-shadow
python src/run_full.py --preset gpu --output runs/gpu_pilot --requests 2026 2027 --seeds 11 23
```

The runner can reuse completed **whole-request** results. It does not resume a partially interrupted request from the last seed; it reruns that request. Results are checkpointed after every model seed inside each request, so partial JSON results are still inspectable.

## 3. Outputs

- `runs/gpu_scalability/results.json`: per-seed and per-method metrics, solver status/witness, held-out flow-region coverage, reference variability, PGD/random comparisons.
- `runs/gpu_memorization/request_<seed>/results.json`: all E6 per-model results, formal witness and solver status, trigger/no-trigger certificates, per-seed reference variability, search comparisons and witness distance.
- `runs/gpu_memorization/request_<seed>/shadow_seed_<seed>.json`: per-target-model balanced shadow attack AUC and low-FPR detection.
- `runs/gpu_memorization/research_report.json`, `REPORT.md`, and PNG figures: aggregated comparisons and conservative request-level Holm-corrected statistics.
- `runs/gpu_memorization/manifest.json`: exact configuration.
- `runs/gpu_memorization/request_<seed>/checkpoints/`: fixed-model state_dict weights for follow-up analysis; never load unknown checkpoints.

## 4. Core scientific claims and boundaries

1. The mixed-integer solver gives exact maximum absolute *logit* disagreement between **fixed, small ReLU models** over a **specified finite latent region** if and only if `status="optimal"`. Timeouts/nonoptimal solves are explicitly **not** certified.
2. The nonlinear triangular coupling flow is invertible and piecewise affine; it is exactly encoded as two affine regions. Calibration uses an L∞ latent score from held-out examples. An L∞ latent box is **not a highest-density level set** of a multivariate Gaussian.
3. E7 verifies a 2D PCA classifier with an artificial trigger feature fixed to 0 or 1, **not** a full 28×28 image CNN, and not a privacy guarantee. The trigger samples are interventions outside the natural image distribution.
4. The matched shadow-model likelihood-ratio score is a *canary-specific membership diagnostic*. This is not formal privacy, differential privacy, or a fully general LiRA implementation.
5. Multiple retraining references quantify *model-level behavioral variability*, not the distributional equivalence required for certified machine unlearning.
6. Holm-corrected sign tests use deletion **requests** as units; repeated seeds for one request are not independent scientific replications. The multi-request runner assigns disjoint canaries using one fixed master shuffle and reports any overlap for auditing; the underlying training sets still overlap strongly.
7. Higher dimensional PCA verification (E5) is separate from the expressive 2D nonlinear-flow benchmark (E7). The package does not claim nonlinear-flow verification for 32D.

## 5. Paper workflow

See `docs/RESEARCH_PROTOCOL.md`, `docs/MANUSCRIPT_RESULTS_TEMPLATE.md`, and `docs/SUBMISSION_CHECKLIST.md`. Report empirical findings only for datasets you have actually run, include all timeout/failure rates, and report disagreement with every retraining reference where evaluated.

## 6. Existing evidence

The `baseline/` directory contains the GPU E5 JSON supplied in the conversation and the earlier local E6 five-seed JSON. `runs/coupling_smoke/` and `runs/scalability_smoke/` are newly executed local smoke experiments. None of these is a substitute for the full E7 MNIST GPU replication.
