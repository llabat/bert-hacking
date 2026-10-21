#!/bin/bash
# One real run on one GPU, then its regressions: checks the whole pipeline before launching
# a job array. The run is a regular one (results/runs/<run_id>/), so it counts as done later.
#
# Submit from the repo root:
#   mkdir -p logs
#   sbatch scripts/smoke_test.sbatch                      # misinfo, config of smoke-test.json
#   sbatch scripts/smoke_test.sbatch config_files/config-loop.json

#SBATCH --job-name=bert-hacking-smoke
#SBATCH --partition=electronic
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm-smoke-%j.out

set -euo pipefail
config=${1:-config_files/config-misinfo.json}

eval "$(micromamba shell hook --shell bash)"
micromamba activate "${BERT_HACKING_ENV:-berthacking}"
cd "$SLURM_SUBMIT_DIR"

# Fail fast if torch cannot use the GPU: driver too old for torch's CUDA build (training would
# silently fall back to CPU), or GPU already held by another process (exclusive mode)
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
python - <<'EOF'
import torch
assert torch.cuda.is_available(), f"CUDA unavailable: torch {torch.__version__} built for CUDA {torch.version.cuda}"
torch.zeros(1, device="cuda")  # creates a context: fails if the GPU is held by another process
print(f"torch {torch.__version__} (CUDA {torch.version.cuda}) on {torch.cuda.get_device_name(0)}, "
      f"bf16 supported: {torch.cuda.is_bf16_supported()}")
EOF

bert-hacking train --config "$config" --subsample config_files/smoke-test.json
bert-hacking regress --config "$config" --workers "${SLURM_CPUS_PER_TASK:-4}"
bert-hacking status --config "$config" --subsample config_files/smoke-test.json
