#!/bin/bash
#SBATCH --job-name=protannot-dapt
#SBATCH --account=CHANGE_ME@h100
#SBATCH -C h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1               # torchrun spawns one process per GPU
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=96
#SBATCH --hint=nomultithread
#SBATCH --time=20:00:00
#SBATCH --qos=qos_gpu_h100-t3
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
module purge
module load arch/h100
module load pytorch-gpu/py3/CHANGE_ME
export HF_HOME=$WORK/hf_cache HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
cd $WORK/projet_protannotml
srun torchrun --standalone --nproc_per_node=4 scripts/dapt_mlm.py \
    --model facebook/esm2_t33_650M_UR50D --train data/ig_train.txt --valid data/ig_valid.txt \
    --out runs/dapt650M --batch 64 --max_steps 60000
