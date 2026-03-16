#!/bin/bash
#SBATCH -J idev6
#SBATCH -o idev6.o%j
#SBATCH -N 2
#SBATCH --ntasks-per-node=1
#SBATCH -p ghtest
#SBATCH -t 00:30:00
#SBATCH -A bddur53
#SBATCH --mail-type=END
#SBATCH --mail-user=nikolai.juraschko@bioch.ox.ac.uk


#------------------------------------------------------
# Source the bashrc to add conda
source /nobackup/projects/bddur53/DDMD_WE_GH_2/conda_folder/miniconda/etc/profile.d/conda.sh
#source ~/.bashrc

# Load the required modules
#ml gcc/14.2.0 cuda/12.5 hdf5
module load gcc/14.2
module load cuda/12.5.1
module load hdf5

# Unset these to prevent conflicts with Parsl's internal srun calls
unset SLURM_CPUS_PER_TASK
unset SLURM_TRES_PER_TASK

conda activate deepdrivewe_VISTA_2

# Change to working directory
#cd /scratch/08288/abrace/projects/ddwe/src/deepdrivewe
cd /nobackup/projects/bddur53/DDMD_WE_GH_2/TRY_2/deepdrive_we-BEDE

# Get the config file for this example
#CONFIG_FILE=/scratch/08288/abrace/projects/ddwe/src/deepdrivewe/examples/openmm_ntl9_ddwe_vista/config.yaml
CONFIG_FILE=/nobackup/projects/bddur53/DDMD_WE_GH_2/TRY_2/deepdrive_we-BEDE/examples/openmm_ntl9_ddwe_vista/config.yaml

# Start a background resource monitor that logs every 10 seconds
(while true; do date; nvidia-smi; free -h; sleep 10; done) > resource_usage.log &
MONITOR_PID=$!

# Run the example
python -m deepdrivewe.examples.openmm_ntl9_ddwe.main --config $CONFIG_FILE

# Kills monitoring after the run finishes
kill $MONITOR_PID
