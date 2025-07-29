#!/bin/bash
#SBATCH -p dllabdlc_gpu-rtx2080
#SBATCH --gpus=1
#SBATCH --mem=20G
#SBATCH --time=24:00:00
#SBATCH -o train_experiment_%j.out
#SBATCH -e train_experiment_%j.err

echo "Starting training experiments"

# Activate conda
source /work/dlclacrge2/celikh-nr1-ayca/miniconda3/etc/profile.d/conda.sh
conda activate rfcl

bash /work/dlclarge2/celikh-nr1-ayca/RFCL-Pytorch/scripts/adroit/adroit_fast.sh