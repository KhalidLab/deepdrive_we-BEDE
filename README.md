# DeepDriveMD on Bede (NVIDIA GH200 Grace Hopper Superchips with NVIDIA NVLink-C2C | aarch64)

See https://github.com/NikJur/DeepDriveMD-BEDE to install DDMD on Bede NVIDIA Tesla V100/IBM POWER9 architecture.

This documentation details the setup required to run DeepDriveMD on the **Bede Supercomputer** (NVIDIA GH200 Grace Hopper Superchips with NVIDIA NVLink-C2C | `aarch64`).

---

## 0. Initial Setup & Conda Installation

### 📍 Recommended Installation Path
On Bede, it is highly recommended to install the source code in your project's `nobackup` directory to avoid storage quotas and ensure fast I/O performance.

**Navigate to your project directory before cloning the GitHub repository (create your user folder if needed):**
```bash
cd /nobackup/projects/<project_code>/<user_name>/
```

**Miniforge (aarch64) installation**\
Miniforge aarch64 provides compatible Conda packages and is required.

```bash
# 1. Create architecture-specific directory:
mkdir -p aarch64
cd aarch64

export CONDADIR=/nobackup/projects/<project>/$USER/aarch64 # Update this with your <project> code.
mkdir -p $CONDADIR
pushd $CONDADIR

# Download the latest miniconda installer for aarch64
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh

# Validate the file checksum matches are listed on https://docs.conda.io/en/latest/miniconda_hashes.html.
sha256sum Miniconda3-latest-Linux-aarch64.sh

# We need to log on to the Grace Hopper login landing node to install Miniconda.
ghlogin -A <project_code> # replace <project_code> with you project billing code

sh Miniconda3-latest-Linux-aarch64.sh -b -p ./miniconda
source miniconda/etc/profile.d/conda.sh
conda update conda -y

# Verify installation:
conda --version # Current version at publishing for future debugging: 'conda 26.1.1'. Your version might differ, which is okay.
```

### DeepDriveMD Installation
Run the following commands:

```bash
module load gcc/14.2
module load cuda/12.5.1
module load hdf5

conda create -n deepdrivewe python=3.12 -y
conda activate deepdrivewe
conda install conda-forge::openmm -y
pip install torch --index-url https://download.pytorch.org/whl/cu124

git clone https://github.com/KhalidLab/deepdrive_we-BEDE.git
cd deepdrive_we-BEDE
pip install -U pip setuptools wheel

# Create a new cache directory in your project folder
mkdir -p /nobackup/projects/<project_code>/<user_name>/aarch64/.pip_cache

# Set environment variables to point there
export PIP_CACHE_DIR="/nobackup/projects/<project_code>/<user_name>/aarch64/.pip_cache"
export TMPDIR="/nobackup/projects/<project_code>/<user_name>/aarch64/.pip_cache"

conda install conda-forge::h5py -y

pip install -e . --no-deps

conda install conda-forge::pyyaml -y
pip install colmena proxystore parsl typer mdtraj mdanalysis scikit-learn mdlearn natsort matplotlib pydantic
```

To run an example on the BEDE Grace Hoppers, update the paths in the submit script
and the YAML config file:
```bash
# Define project and user credentials
PROJECT_CODE="<your_project_code>" #replace "<your_project_code>" in this line.
USER_NAME="$(whoami)"

# Define files to be updated
SUBMIT_SCRIPT="examples/openmm_ntl9_ddwe_vista/submit.sh"
CONFIG_FILE="examples/openmm_ntl9_ddwe_vista/config.yaml"

# Perform global replacement of placeholder paths
sed -i "s|<project_code>/<user_name>|${PROJECT_CODE}/${USER_NAME}|g" "$SUBMIT_SCRIPT"  # copy-paste as is! Do NOT replace anything in this line.
sed -i "s|<project_code>/<user_name>|${PROJECT_CODE}/${USER_NAME}|g" "$CONFIG_FILE"    # copy-paste as is! Do NOT replace anything in this line.

echo "Paths updated successfully for user ${USER_NAME} in project ${PROJECT_CODE}."
```

Then run the following command:
```bash
sbatch examples/openmm_ntl9_ddwe_vista/submit.sh
```

## 📂 X. Directory Structure

Your directory should look something like this:

```text
/nobackup/projects/<project_code>/<user_name>/aarch64/
├── deepDrivewe-BEDE/       # This repository
│   └── examples/      # Contains example run files
│       └── openmm_ntl9_ddwe_vista/
│          ├── config.yaml
│          └── submit.sh
└── miniconda/               # Conda installation and later environments
```

## Usage
```bash
# We need to log on to the Grace Hopper login landing node to submit a job to the GH nodes.
ghlogin -A <project_code> # replace <project_code> with you project billing code
```

The main files you will want to edit for your simulations (other than pdb and toplogy files) are submit.sh, config.yaml, and cvae-config.yaml.

To check if any errors occurred in simulations or inference after you job finishes:
```bash
cat runs/ntl9-v1/result/inference.json | grep '"success": false'
cat runs/ntl9-v1/result/simulation.json | grep '"success": false'
```

To check the number of iterations completed:
```bash
h5ls -d runs/ntl9-v1/west.h5/iterations
```
In our ntl9-v1 example, you should see the following output: \
'''iter_00000001            Group \
iter_00000002            Group \
iter_00000003            Group \
iter_00000004            Group \
iter_00000005            Group \
iter_00000006            Group \
iter_00000007            Group \
iter_00000008            Group \
iter_00000009            Group \
iter_00000010            Group
'''

Further information on running with SynD and OpenMM is available from https://github.com/ramanathanlab/deepdrivewe.
