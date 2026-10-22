#!/bin/bash
# Submit N dependent jobs: each one resumes from the last checkpoint of the previous one.
# Usage: bash slurm/chain.sh slurm/h100_train.slurm 3
set -euo pipefail
script=$1; n=${2:-3}
jid=$(sbatch --parsable "$script"); echo "submitted $jid"
for i in $(seq 2 "$n"); do
  jid=$(sbatch --parsable --dependency=afterany:"$jid" "$script"); echo "submitted $jid (after previous)"
done
