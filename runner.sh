#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:a100
#SBATCH --job-name=torch_test_dawnbench
#SBATCH -o torch_test_dawnbench.out
conda activate torch
python CIFAR10.py