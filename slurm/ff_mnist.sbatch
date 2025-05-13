#!/bin/bash
#
#SBATCH --job-name=mnist_ff
#SBATCH --output=../logs/mnist/mnist_ff_%A.log
#SBATCH --error=../logs/mnist/mnist_ff_error_%A.log
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=10:00:00
#SBATCH --partition=ampere24
#SBATCH --gres=gpu:a30:1

echo "Starting MNIST FF job..."

# Activate Virtual Environment
source ~/discovery_project/venv/bin/activate

# Run Python Script
python3 ../scripts/mnist_ff_cnn.py

echo "Job complete."

