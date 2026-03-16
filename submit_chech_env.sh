#!/bin/bash
#SBATCH -J env_check           # Job name
#SBATCH -o check_%j.out        # output file
#SBATCH -e check_%j.err        # error file
#SBATCH -p ghtest              # Use the Grace Hopper test partition
#SBATCH -N 1                   # Request exactly 1 node
#SBATCH --ntasks-per-node=1    # Run 1 task
#SBATCH --gres=gpu:1           # Request 1 GPU (essential for CUDA check)
#SBATCH -t 00:05:00
#SBATCH -A bddur53

# 1. Source the base conda configuration
source /nobackup/projects/bddur53/DDMD_WE_GH_2/conda_folder/miniconda/etc/profile.d/conda.sh

# 2. Load necessary cluster modules for aarch64
module purge
module load gcc/14.2
module load cuda/12.5.1

# 3. Activate your environment
conda activate deepdrivewe_VISTA_2

# 4. Run the check script
echo "Starting Environment Check on $(hostname) at $(date)"
python check_env.py
echo "Check Completed at $(date)"
