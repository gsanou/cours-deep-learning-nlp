#!/bin/bash
#SBATCH --job-name=protannot-v100
#SBATCH --account=CHANGE_ME@v100
#SBATCH -C v100-32g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10                # 40 cores / 4 GPUs
#SBATCH --hint=nomultithread
#SBATCH --time=20:00:00
#SBATCH --qos=qos_gpu-t3
#SBATCH --signal=USR1@300
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
module purge
module load pytorch-gpu/py3/CHANGE_ME
export HF_HOME=$WORK/hf_cache HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
cd $WORK/projet_protannotml
srun python scripts/train_multitask.py --config configs/v100.yaml
