#!/bin/bash
#SBATCH -J DDMD_test
#SBATCH -o DDMD_test.o%j
#SBATCH -N 2
#SBATCH --ntasks-per-node=1
#SBATCH -p ghtest
#SBATCH -t 00:30:00
#SBATCH -A <project_billing_code>

#------------------------------------------------------
# Source conda
source /nobackup/projects/<project_code>/<user_name>/aarch64/miniconda/etc/profile.d/conda.sh
# Alternatively, add it to your bashrc and source ~/.bashrc

# Load the required modules
module load gcc/14.2
module load cuda/12.5.1
module load hdf5

# Unset these to prevent conflicts with Parsl's internal srun calls
unset SLURM_CPUS_PER_TASK
unset SLURM_TRES_PER_TASK

conda activate deepdrivewe

# Change to working directory
cd /nobackup/projects/<project_code>/<user_name>/aarch64/deepdrive_we-BEDE

# Get the config file for this example
CONFIG_FILE=/nobackup/projects/<project_code>/<user_name>/aarch64/deepdrive_we-BEDE/examples/openmm_ntl9_ddwe_vista/config.yaml

# Start a background resource monitor that logs every 10 seconds
(while true; do date; nvidia-smi; free -h; sleep 10; done) > resource_usage.log &
MONITOR_PID=$!

# Run the example
python -m deepdrivewe.examples.openmm_ntl9_ddwe.main --config $CONFIG_FILE

# Kills monitoring after the run finishes
kill $MONITOR_PID
