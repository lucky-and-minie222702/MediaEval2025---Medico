#!/bin/bash
#SBATCh --account=nhdang
#SBATCH --job-name=triet_ptnk

#SBATCH --output=slurm_output/output_%j.log
#SBATCH --error=slurm_output/error_%j.log

#SBATCH --ntasks=1

#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1 

#SBATCH --nodelist=gpu01

source ~/.bashrc
conda activate triet_ptnk
python train.py epochs 50