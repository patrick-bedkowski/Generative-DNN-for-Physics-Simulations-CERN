#!/bin/bash
#SBATCH --job-name=ex-2
#SBATCH --time=16:00:00
#SBATCH --account=plgdynamic2-gpu-a100
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu

module add CUDA/11.8.0

source ../../myenv/bin/activate

python -u ~/pytorch/dynamic_router/expertsim_neutron_2_001_00001.py

echo "fin"
