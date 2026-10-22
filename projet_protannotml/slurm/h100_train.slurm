#!/bin/bash
#SBATCH --job-name=protannot-h100
#SBATCH --account=CHANGE_ME@h100          # your project, check with: idracct
#SBATCH -C h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4               # one task per GPU
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=24                # 96 cores / 4 GPUs
#SBATCH --hint=nomultithread
#SBATCH --time=20:00:00
#SBATCH --qos=qos_gpu_h100-t3
#SBATCH --signal=USR1@300                 # SIGUSR1 to every task 5 min before the time limit
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
module purge
module load arch/h100                      # must come first on the H100 partition
module load pytorch-gpu/py3/CHANGE_ME      # pick a version: module avail pytorch-gpu
export HF_HOME=$WORK/hf_cache HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1) MASTER_PORT=29500
cd $WORK/projet_protannotml
srun python scripts/train_multitask.py --config configs/h100.yaml
