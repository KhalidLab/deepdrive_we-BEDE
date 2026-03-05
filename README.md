# NOTES for BEDE:
- AMBER (including AmberTools) is only provided on ppc64le partitions/nodes (i.e. Power9) but NOT aarch64 (i.e. GraceHoppers).



# deepdrivewe
Implementation of [WESTPA](https://westpa.github.io/westpa/index.html) using [Colmena](https://github.com/exalearn/colmena/tree/master).

## Installation

To install the package, run the following command:
```bash
git clone https://github.com/NikJur/deepdrive_we-BEDE.git
cd deepdrivewe
pip install -e .
```

Full installation including dependencies:
```bash
git clone https://github.com/NikJur/deepdrive_we-BEDE.git
cd deepdrivewe
#old: conda create -n deepdrivewe python=3.10 -y
conda create -n deepdrivewe python=3.12 -y
#old: conda install omnia::ambertools -y#
conda install dacase::ambertools-dac=25 #needs python 3.12#

#To enable CUDA support, UCX requires the CUDA Runtime library (libcudart).
#The library can be installed with the appropriate command below:
#* For CUDA 12, run:    conda install cuda-cudart cuda-version=12
#* For CUDA 13, run:    conda install cuda-cudart cuda-version=13
#If any of the packages you requested use CUDA then CUDA should already
#have been installed for you.
#To enable CUDA support, please follow UCX's instruction above.
#To additionally enable NCCL support, run:    conda install nccl
#On Linux, Open MPI is built with CUDA awareness but it is disabled by default.
#To enable it, please set the environment variable
#OMPI_MCA_opal_cuda_support=true
#before launching your MPI processes.
#Equivalently, you can set the MCA parameter in the command line:
#mpiexec --mca opal_cuda_support 1 ...
#Note that you might also need to set UCX_MEMTYPE_CACHE=n for CUDA awareness via
#UCX. Please consult UCX documentation for further details.

#old: conda install conda-forge::openmm==7.7 -y
conda install conda-forge::openmm -y

pip install -e .

#  Building wheel for h5py (pyproject.toml) ... error
#  error: subprocess-exited-with-error
#  × Building wheel for h5py (pyproject.toml) did not run successfully.
#  │ exit code: 1
#  ╰─> [305 lines of output]
#      /tmp/pip-build-env-njmmdowk/overlay/lib/python3.12/site-packages/setuptools/config/_apply_pyprojecttoml.py:82: SetuptoolsDeprecationWarning: `project.license` as a TOML table is deprecated
```
#To use deep learning models, install the correct version of [PyTorch](https://pytorch.org/get-started/locally/)
#for your system and drivers. To use `mdlearn`, you may need an earlier version of PyTorch:
```bash
pip install torch==1.12
```

### Installation on VISTA

To install the package on VISTA, run the following commands:
```bash
#old: ml gcc/14.2.0 cuda/12.5 hdf5
module load gcc/14.2
module load cuda/12.5.1
module load hdf5

conda create -n deepdrivewe python=3.12 -y
conda activate deepdrivewe
conda install conda-forge::openmm -y
pip install torch --index-url https://download.pytorch.org/whl/cu124

git clone git@github.com:braceal/deepdrivewe.git
cd deepdrivewe
pip install -U pip setuptools wheel

# Create a new cache directory in your project folder
mkdir -p /nobackup/projects/bddur53/DDMD_WE_GH_2/.pip_cache

# Set environment variables to point there
export PIP_CACHE_DIR="/nobackup/projects/bddur53/DDMD_WE_GH_2/.pip_cache"
export TMPDIR="/nobackup/projects/bddur53/DDMD_WE_GH_2/.pip_cache"

pip install -e .
#if error "Failed building wheel for h5py":
conda install conda-forge::h5py -y
conda install conda-forge::pyyaml -y
```
To run an example on VISTA, update the absolute paths in the submit script
and the YAML config file, and then run the following command:
```bash
sbatch examples/openmm_ntl9_ddwe_vista/submit.sh
```

## Usage
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
