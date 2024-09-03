#!/bin/bash
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:a100
#SBATCH --job-name=torch_dawnbench
#SBATCH -o torch_dawnbench.out
conda activate torch
python CIFAR10.py