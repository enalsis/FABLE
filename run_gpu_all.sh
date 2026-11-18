#!/usr/bin/env bash
# Full research validation. GPU for PyTorch training; SciPy/HiGHS MILP uses CPU.
set -euo pipefail
cd "$(dirname "$0")"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}" OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
python -m unittest discover -s tests -v
python - <<'PY'
import torch
print('torch:',torch.__version__, 'CUDA available:',torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit('CUDA is required for full GPU protocol; run run_smoke.sh for CPU validation')
print('GPU:',torch.cuda.get_device_name(0))
PY

# A: higher-dimensional 2/8/16/32 PCA with negative controls, retrain variability,
# membership diagnostics, actual MILP statuses and PGD/random baselines.
python src/experiment_e5.py \
  --dataset mnist --data-dir ./data --download --device cuda --output runs/gpu_scalability \
  --dimensions 2 8 16 32 \
  --seeds 11 23 37 51 71 83 97 109 127 149 \
  --selection-seed 2026 --forget-n 200 --poison --canary-fraction 0.2 \
  --hidden 12 --train-steps 1800 --short-steps 100 --long-steps 700 \
  --lr 0.008 --unlearn-lr 0.003 --batch 128 --ascent-weight 0.15 --n-retrains 3 \
  --coverages 0.5 0.8 0.95 --timeout 120 \
  --counterexample-methods original reset_head fine_tune_long --counterexample-q 0.8 \
  --random-samples 5000 --pgd-restarts 32 --pgd-steps 100

python src/analyze_scalability.py --results runs/gpu_scalability/results.json

# B: memorization-grounded experiment with controlled canaries, sham deletion,
# nonlinear flow, 5 deletion requests * 10 seeds, 3 retrains, and 16 shadow models
# per seed. This is substantially more expensive than the CPU smoke test.
python src/run_full.py --preset gpu --output runs/gpu_memorization

echo 'COMPLETE: runs/gpu_scalability and runs/gpu_memorization'
