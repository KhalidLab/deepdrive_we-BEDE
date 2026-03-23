# DeepDriveMD-WE on Bede (NVIDIA GH200 Grace Hopper Superchips | aarch64)

This documentation details the setup required to run DeepDriveMD-WE on the Bede Supercomputer using the NVIDIA GH200 Grace Hopper nodes ('aarch64' architecture).

For installation on Bede NVIDIA Tesla V100/IBM POWER9 architecture, see https://github.com/NikJur/DeepDriveMD-BEDE.

---

## 0. Initial Setup & Conda Installation

### 📍 Recommended Installation Path
On Bede, it is highly recommended to install the source code in your project's `nobackup` directory to avoid storage quotas and ensure fast I/O performance.

**Navigate to your project directory (create your user folder if needed):**
```bash
# Replace <project_code> with your project code
cd /nobackup/projects/<project_code>/$(whoami)/

mkdir -p aarch64
cd aarch64
```

**Miniforge (aarch64) installation**\
Miniforge aarch64 provides compatible Conda packages and is required.

```bash
export CONDADIR=/nobackup/projects/<project_code>/$(whoami)/aarch64 # Update this with your <project_code>.
mkdir -p $CONDADIR
pushd $CONDADIR

# Download the latest miniconda installer for aarch64
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh

# Validate the file checksum matches are listed on https://docs.conda.io/en/latest/miniconda_hashes.html.
sha256sum Miniconda3-latest-Linux-aarch64.sh

# Log in to the Grace Hopper landing node to perform the installation
ghlogin -A <project_code> # replace <project_code> with your project billing code

# Install Miniconda to the current directory and update conda
sh Miniconda3-latest-Linux-aarch64.sh -b -p ./miniconda
source miniconda/etc/profile.d/conda.sh
conda update conda -y

# Verify installation:
conda --version # Current version at publishing for future debugging: 'conda 26.1.1'. Your version might differ, which is okay.
```

### 1. DeepDriveMD-WE Installation
Configure the environment and install dependencies for the H200 accelerators and NVLink-C2C interconnect.
Run the following commands:

```bash
# Load required system modules
module load gcc/14.2
module load cuda/12.5.1
module load hdf5

# Create and activate the deepdrive conda environment
conda create -n deepdrivewe python=3.12 -y
conda activate deepdrivewe

# Install core MD and AI libraries
conda install conda-forge::openmm -y
pip install torch --index-url https://download.pytorch.org/whl/cu124

# Clone the BEDE-specific DeepDriveMD repository
git clone https://github.com/KhalidLab/deepdrive_we-BEDE.git
cd deepdrive_we-BEDE
pip install -U pip setuptools wheel

# Create a new cache directory in your project folder
mkdir -p /nobackup/projects/<project_code>/$(whoami)/aarch64/.pip_cache

# Configure persistent cache to stay within nobackup
export PIP_CACHE_DIR="/nobackup/projects/<project_code>/$(whoami)/aarch64/.pip_cache"
export TMPDIR="/nobackup/projects/<project_code>/$(whoami)/aarch64/.pip_cache"

# Install more dependencies and DeepDriveMD itself (follow the order given)
conda install conda-forge::h5py -y
pip install -e . --no-deps
conda install conda-forge::pyyaml -y
pip install colmena proxystore parsl typer mdtraj mdanalysis scikit-learn mdlearn natsort matplotlib pydantic
```

### 2. Running the NTL9 Example
To run an example on the BEDE Grace Hoppers, use the following automation block to synchronise paths in your submission script and YAML configuration:

```bash
# Define project and user credentials
PROJECT_CODE="<your_project_code>" #replace "<your_project_code>" in this line.
USER_NAME="$(whoami)"

# Define files to be updated
SUBMIT_SCRIPT="examples/openmm_ntl9_ddwe_vista/submit.sh"
CONFIG_FILE="examples/openmm_ntl9_ddwe_vista/config.yaml"

# Perform global replacement of placeholder paths
sed -i "s|<project_code>/<user_name>|${PROJECT_CODE}/${USER_NAME}|g" "$SUBMIT_SCRIPT"  # copy-paste as is! Do NOT replace anything in this line.
sed -i "s|<project_billing_code>|${PROJECT_CODE}|g" "$SUBMIT_SCRIPT"  # copy-paste as is! Do NOT replace anything in this line.
sed -i "s|<project_code>/<user_name>|${PROJECT_CODE}/${USER_NAME}|g" "$CONFIG_FILE"    # copy-paste as is! Do NOT replace anything in this line.

echo "Paths updated successfully for user ${USER_NAME} in project ${PROJECT_CODE}."
```

Then run the following command to submit the job for production on the ghtest partition:
```bash
sbatch examples/openmm_ntl9_ddwe_vista/submit.sh
```

## 📂 Directory Structure

A successful installation will result in the following layout:

```text
/nobackup/projects/<project_code>/<user_name>/aarch64/
├── deepDrivewe-BEDE/            # DDMD repository
│   └── examples/                # Contains example run files
│       └── openmm_ntl9_ddwe_vista/
│           ├── config.yaml      # MD simulation workflow config
│           ├── cvae-config.yaml # ML model hyper-parameters
│           └── submit.sh        # Slurm submission script
└── miniconda/                   # Conda installation and environments
```

## Usage
The primary files for configuration (other than pdb and topology files) are 'submit.sh', 'config.yaml', and 'cvae-config.yaml'. I suggest you start by looking at those to get started on your first production run, post example run.

To check if any errors occurred in simulations or inference after your job completed:
```bash
cat runs/ntl9-v1/result/inference.json | grep '"success": false'
cat runs/ntl9-v1/result/simulation.json | grep '"success": false'
```

To check the number of iterations completed:
```bash
h5ls -d runs/ntl9-v1/west.h5/iterations
```

In our ntl9-v1 example, you should see the following output:
```bash
iter_00000001            Group
iter_00000002            Group
iter_00000003            Group
iter_00000004            Group
iter_00000005            Group
iter_00000006            Group
iter_00000007            Group
iter_00000008            Group
iter_00000009            Group
iter_00000010            Group
```

```bash
# Every time we want to submit a job to the Grace Hopper nodes, we need to log on to the Grace Hopper login landing node first. This holds true for both the 'ghtest' and 'gh' partition (see submit.sh and BEDE documentation).
ghlogin -A <project_code> # replace <project_code> with you project billing code
```

Further information on running DeepDriveMD with SynD and OpenMM is available from https://github.com/ramanathanlab/deepdrivewe.
