#!/bin/bash
# Train every config of an experiment with a SLURM job array: one GPU per array task, each
# task running a disjoint shard of the configs. Tasks can stop and resume at any time: a run
# is done when results/runs/<run_id>/run.json exists, and an interrupted run is redone.
#
# Submit from the repo root (logs/ must exist before submitting):
#   mkdir -p logs
#   sbatch --array=0-7 scripts/train.sbatch config_files/config-misinfo.json
#   sbatch --array=0-7 scripts/train.sbatch config_files/config-misinfo.json --subsample config_files/sample-configurations.json
# Pin a GPU model with e.g. --gres=gpu:A6000:1 on the sbatch command line.
# The array indices must be 0..n-1: task i runs shard i/n.

#SBATCH --job-name=bert-hacking
#SBATCH --partition=hard
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/slurm-%A_%a.out

set -euo pipefail
config=${1:?usage: sbatch --array=0-N scripts/train.sbatch CONFIG [train options]}
shift

eval "$(micromamba shell hook --shell bash)"
micromamba activate "${BERT_HACKING_ENV:-berthacking}"
cd "$SLURM_SUBMIT_DIR"

bert-hacking train --config "$config" \
    --shard "${SLURM_ARRAY_TASK_ID:-0}/${SLURM_ARRAY_TASK_COUNT:-1}" "$@"
