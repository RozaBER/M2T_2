#!/bin/bash
#SBATCH --job-name=brain2text
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --mem=64G
#SBATCH --partition=gpu

# Load modules (adjust based on your cluster)
module load python/3.8
module load cuda/11.8

# Activate virtual environment if using one
# source venv/bin/activate

# Set environment variables
export CUDA_VISIBLE_DEVICES=0,1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Run training
./setup_and_run.sh