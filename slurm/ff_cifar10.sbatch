#!/bin/bash
#
#SBATCH --job-name=cifar10_ff       # Job Name
#SBATCH --output=../logs/cifar10/cifar10_ff_%A.log  # Log File
#SBATCH --error=../logs/cifar10/cifar10_ff_error_%A.log  # Error Log File
#
#SBATCH --ntasks=1                   # Number of tasks (single job)
#SBATCH --cpus-per-task=4             # Number of CPU cores per task
#SBATCH --time=10:00:00               # Max execution time (10 hours)
#SBATCH --partition=ampere24          # Specify GPU partition
#SBATCH --gres=gpu:a30:1             # Request 1 GPU (modify if needed)

echo "Starting CIFAR-10 FF job..."

# Load necessary modules (modify as needed)
module load python/3.9  # Adjust version if necessary

# Activate Virtual Environment
source ~/discovery_project/venv/bin/activate

# Run Python Script
python3 ../scripts/cifar10_ff.py

echo "Job complete."

