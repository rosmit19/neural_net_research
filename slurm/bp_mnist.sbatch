#!/bin/bash
#
#SBATCH --job-name=mnist_bp  # Set an appropriate job name
#SBATCH --output=../logs/mnist/mnist_bp_%A.log  # Log file
#SBATCH --error=../logs/mnist/mnist_bp_error_%A.log  # Error log file
#
#SBATCH --ntasks=1  # Number of tasks (single job)
#SBATCH --cpus-per-task=2  # Number of CPUs per task
#SBATCH --time=10:00:00  # Maximum execution time (10 hours)
#SBATCH --partition=ampere24  # Change this based on your HPC cluster's partition
#SBATCH --gres=gpu:a30:1  # Request 1 GPU (modify if necessary)

echo "Starting MNIST BP job..."

# Activate Virtual Environment
source ~/discovery_project/venv/bin/activate

# Run Python Script
python3 ../scripts/mnist_bp_cnn.py

echo "Job complete."

