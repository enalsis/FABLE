#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}" OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
python -m unittest discover -s tests -v
python src/run_full.py --preset smoke --output runs/smoke
python src/experiment_e5.py --dataset digits --device cpu --output runs/scalability_smoke --dimensions 2 8 16 --seeds 11 --forget-n 24 --poison --hidden 6 --train-steps 80 --short-steps 12 --long-steps 25 --timeout 15 --random-samples 100 --pgd-restarts 2 --pgd-steps 8

python src/analyze_scalability.py --results runs/scalability_smoke/results.json
