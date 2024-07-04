#!/bin/bash
#SBATCH --job-name="JupyterGPU"
#SBATCH --output=jupyter-log-%J.txt
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1  # Request one GPU
#SBATCH --time=5:00:00  # Request runtime
#SBATCH --mem=32G  # Request 16GB of system memory
#SBATCH --cpus-per-task=4  # Request 4 CPUs

# Activate your conda environment
source activate dl_a2

# Find an open port to start Jupyter on (avoid conflicts)
export PORT=$(shuf -i 8000-9999 -n 1)
echo "Jupyter is running on port $PORT"

# Start the Jupyter notebook
jupyter notebook --no-browser --ip=$(hostname -f) --port=$PORT

