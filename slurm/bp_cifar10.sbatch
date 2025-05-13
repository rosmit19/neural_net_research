#!/bin/bash
#
#SBATCH --job-name=cifar10_bp       # Job Name
#SBATCH --output=../logs/cifar10/cifar10_bp_%A.log  # Log File
#SBATCH --error=../logs/cifar10/cifar10_bp_error_%A.log  # Error Log File
#
#SBATCH --ntasks=1                   # Number of tasks (single job)
#SBATCH --cpus-per-task=4             # Number of CPU cores per task
#SBATCH --time=10:00:00               # Max execution time (10 hours)
#SBATCH --partition=ampere24          # Specify GPU partition
#SBATCH --gres=gpu:a30:1             # Request 1 GPU (modify if needed)

echo "Starting CIFAR-10 BP job..."

# Load necessary modules (modify as needed)
module load python  # Adjust version if necessary

# Activate Virtual Environment
source ~/discovery_project/venv/bin/activate

# Run Python Script
python ../scripts/cifar10_bp.py

echo "Job complete."

