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

To run the example, run the following command:
```bash
python -m deepdrivewe.examples.amber_hk.main --config examples/amber_nacl_hk/config.yaml
```

To kill all the workers, run the following command:
```bash
ps -e | grep -E 'sander|python|process_worker|parsl' | awk '{print $1}' | xargs kill
```

To check if any errors occurred in simulations or inference:
```bash
cat runs/naive_resampler_test_v2/result/inference.json | grep '"success": false'
cat runs/naive_resampler_test_v2/result/simulation.json | grep '"success": false'
```

To check the number of iterations completed:
```bash
h5ls -d runs/naive_resampler_test_v2/west.h5/iterations
```

### Running with SynD
To use the SynD simulation engine, install the following dependencies:
```bash
pip install git+https://github.com/jeremyleung521/SynD.git@rng-fix
```

To generate the basis state .npy files from a .txt file, run the following command:
```bash
python -m deepdrivewe.simulation.synd --basis-states examples/synd_ntl9/bstates.txt --output-dir examples/synd_ntl9/bstates
```

To run the example, run the following command:
```bash
nohup python -m deepdrivewe.examples.synd_ntl9.main --config examples/synd_ntl9/config.yaml &> nohup.log &
```

### Running with OpenMM
To run the example, run the following command:
```bash
OPENMM_CPU_THREADS=1 nohup python -m deepdrivewe.examples.openmm_ntl9_hk.main --config examples/openmm_ntl9_hk/config.yaml &> nohup.log &
```

Note that we set `OPENMM_CPU_THREADS=1` to restrict each OpenMM simulation to a single thread. This is necessary to prevent
the simulations from using all available CPU resources. You can also run the simulations on a GPU by adjusting the Parsl configuration.

## Contributing

For development, it is recommended to use a virtual environment. The following
commands will create a virtual environment, install the package in editable
mode, and install the pre-commit hooks.
```bash
python -m venv venv
source venv/bin/activate
pip install -U pip setuptools wheel
pip install -e '.[dev,docs]'
pre-commit install
```
To test the code, run the following command:
```bash
pre-commit run --all-files
tox -e py310
```
