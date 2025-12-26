#!/bin/bash
#SBATCH --job-name=multiref_b2t
#SBATCH --output=output_multiref_%j.log
#SBATCH --error=error_multiref_%j.log
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=kingspeak-gpu
#SBATCH --account=kingspeak-gpu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=id@utah.edu

# ==============================================
# Multi-Reference CTC Training for Brain-to-Text
# Using COT_CAUGHT merger (AA ↔ AO)
# ==============================================

echo "=============================================="
echo "Job Information"
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "Current directory: $(pwd)"
echo ""

# Load required modules (adjust for your cluster)
# module load python/3.10
# module load cuda/11.8

# Activate virtual environment if using one
# source ~/venvs/b2t/bin/activate

# Print environment info
echo "=============================================="
echo "Environment Information"
echo "=============================================="
echo "Python: $(which python)"
echo "Python version: $(python --version)"
nvidia-smi
echo ""

# Change to model_training directory
cd model_training

echo "=============================================="
echo "Starting Multi-Reference Training"
echo "=============================================="
echo "Config: rnn_args_multiref.yaml"
echo "Mergers: COT_CAUGHT (AA ↔ AO)"
echo ""

# Run training
python -u train_model_multiref.py --config rnn_args_multiref.yaml

echo ""
echo "=============================================="
echo "Training Complete"
echo "=============================================="
echo "End time: $(date)"
