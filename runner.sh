#!/bin/bash
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:a100
#SBATCH --job-name=train_GAN
#SBATCH -o Result_GAN.out
conda activate demo
python runGAN.py