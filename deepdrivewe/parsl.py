"""Utilities to build Parsl configurations."""

from __future__ import annotations

import sys
from abc import ABC
from abc import abstractmethod
from pathlib import Path
from typing import Literal
from typing import Sequence
from typing import Union

if sys.version_info >= (3, 11):  # pragma: >=3.11 cover
    from typing import Self
else:  # pragma: <3.11 cover
    from typing_extensions import Self


from parsl.config import Config
from parsl.executors import HighThroughputExecutor
from parsl.launchers import WrappedLauncher, SimpleLauncher
from parsl.providers import LocalProvider
from pydantic import BaseModel
from pydantic import Field
from pydantic import model_validator

import logging
from parsl.addresses import address_by_interface
# Silence noisy Parsl scaling and strategy logs
logging.getLogger('parsl.jobs.strategy').setLevel(logging.WARNING)
logging.getLogger('parsl.executors.high_throughput.zmq_pipes').setLevel(logging.WARNING)
logging.getLogger('parsl.executors.high_throughput.interchange').setLevel(logging.WARNING)


class BaseComputeConfig(BaseModel, ABC):
    """Compute config (HPC platform, number of GPUs, etc)."""

    # Name of the platform to uniquely identify it
    name: Literal[''] = ''

    @abstractmethod
    def get_parsl_config(self, run_dir: str | Path) -> Config:
        """Create a new Parsl configuration.

        Parameters
        ----------
        run_dir : str | Path
            Path to store monitoring DB and parsl logs.

        Returns
        -------
        Config
            Parsl configuration.
        """
        ...


class LocalConfig(BaseComputeConfig):
    """Local compute config."""

    name: Literal['local'] = 'local'  # type: ignore[assignment]

    max_workers_per_node: int = Field(
        default=1,
        description='Number of workers to use.',
    )
    cores_per_worker: float = Field(
        default=1.0,
        description='Number of cores per worker.',
    )
    worker_port_range: tuple[int, int] = Field(
        default=(10000, 20000),
        description='Port range for the workers.',
    )
    label: str = Field(
        default='cpu_htex',
        description='Label for the executor.',
    )

    def get_parsl_config(self, run_dir: str | Path) -> Config:
        """Generate a Parsl configuration for local execution."""
        return Config(
            run_dir=str(run_dir),
            strategy=None,
            executors=[
                HighThroughputExecutor(
                    address='localhost',
                    label=self.label,
                    max_workers_per_node=self.max_workers_per_node,
                    cores_per_worker=self.cores_per_worker,
                    worker_port_range=self.worker_port_range,
                    provider=LocalProvider(init_blocks=1, max_blocks=1),
                ),
            ],
        )


class WorkstationConfig(BaseComputeConfig):
    """Compute config for a workstation."""

    name: Literal['workstation'] = 'workstation'  # type: ignore[assignment]

    available_accelerators: int | Sequence[str] = Field(
        default=1,
        description='Number of GPU accelerators to use.',
    )
    worker_port_range: tuple[int, int] = Field(
        default=(10000, 20000),
        description='Port range for the workers.',
    )
    retries: int = Field(
        default=1,
        description='Number of retries for the task.',
    )
    label: str = Field(
        default='gpu_htex',
        description='Label for the executor.',
    )

    def get_parsl_config(self, run_dir: str | Path) -> Config:
        """Generate a Parsl configuration for workstation execution."""
        return Config(
            run_dir=str(run_dir),
            retries=self.retries,
            executors=[
                HighThroughputExecutor(
                    address='localhost',
                    label=self.label,
                    cpu_affinity='block',
                    available_accelerators=self.available_accelerators,
                    worker_port_range=self.worker_port_range,
                    provider=LocalProvider(init_blocks=1, max_blocks=1),
                ),
            ],
        )


class HybridWorkstationConfig(BaseComputeConfig):
    """Run simulations on CPU and AI models on GPU."""

    name: Literal['hybrid_workstation'] = 'hybrid_workstation'  # type: ignore[assignment]

    cpu_config: LocalConfig = Field(
        description='Config for the CPU executor to run simulations.',
    )
    gpu_config: WorkstationConfig = Field(
        description='Config for the GPU executor to run AI models.',
    )

    def get_parsl_config(self, run_dir: str | Path) -> Config:
        """Generate a Parsl configuration for hybrid execution."""
        return Config(
            run_dir=str(run_dir),
            retries=self.gpu_config.retries,
            executors=[
                HighThroughputExecutor(
                    address='localhost',
                    label=self.cpu_config.label,
                    max_workers_per_node=self.cpu_config.max_workers_per_node,
                    cores_per_worker=self.cpu_config.cores_per_worker,
                    worker_port_range=self.cpu_config.worker_port_range,
                    provider=LocalProvider(init_blocks=1, max_blocks=1),
                ),
                HighThroughputExecutor(
                    address='localhost',
                    label=self.gpu_config.label,
                    cpu_affinity='block',
                    available_accelerators=self.gpu_config.available_accelerators,
                    worker_port_range=self.gpu_config.worker_port_range,
                    provider=LocalProvider(init_blocks=1, max_blocks=1),
                ),
            ],
        )


class InferenceTrainWorkstationConfig(BaseComputeConfig):
    """Run simulations on CPU and AI models on GPU."""

    name: Literal[
        'inference_train_workstation'
    ] = 'inference_train_workstation'  # type: ignore[assignment]

    cpu_config: LocalConfig = Field(
        description='Config for the CPU executor to run simulations.',
    )
    train_gpu_config: WorkstationConfig = Field(
        description='Config for the GPU executor to run AI models.',
    )
    inference_gpu_config: WorkstationConfig = Field(
        description='Config for the GPU executor to run AI models.',
    )

    @model_validator(mode='after')
    def validate_htex_labels(self) -> Self:
        """Ensure that the labels are unique."""
        self.cpu_config.label = 'simulation_htex'
        self.train_gpu_config.label = 'train_htex'
        self.inference_gpu_config.label = 'inference_htex'
        return self

    def get_parsl_config(self, run_dir: str | Path) -> Config:
        """Generate a Parsl configuration for hybrid execution."""
        return Config(
            run_dir=str(run_dir),
            retries=self.train_gpu_config.retries,
            executors=[
                HighThroughputExecutor(
                    address='localhost',
                    label=self.cpu_config.label,
                    max_workers_per_node=self.cpu_config.max_workers_per_node,
                    cores_per_worker=self.cpu_config.cores_per_worker,
                    worker_port_range=self.cpu_config.worker_port_range,
                    provider=LocalProvider(init_blocks=1, max_blocks=1),
                ),
                HighThroughputExecutor(
                    address='localhost',
                    label=self.train_gpu_config.label,
                    cpu_affinity='block',
                    available_accelerators=self.train_gpu_config.available_accelerators,
                    worker_port_range=self.train_gpu_config.worker_port_range,
                    provider=LocalProvider(init_blocks=1, max_blocks=1),
                ),
                HighThroughputExecutor(
                    address='localhost',
                    label=self.inference_gpu_config.label,
                    cpu_affinity='block',
                    available_accelerators=self.inference_gpu_config.available_accelerators,
                    worker_port_range=self.inference_gpu_config.worker_port_range,
                    provider=LocalProvider(init_blocks=1, max_blocks=1),
                ),
            ],
        )


class VistaConfig(BaseComputeConfig):
    """VISTA compute config.

    https://tacc.utexas.edu/systems/vista/
    """

    name: Literal['vista'] = 'vista'  # type: ignore[assignment]

    num_nodes: int = Field(
        ge=2,
        description='Number of nodes to use (must use at least 3 nodes).',
    )

    # We have a long idletime to ensure train/inference executors are not
    # shut down (to enable warmstarts) while simulations are running.
    max_idletime: float = Field(
        default=60.0 * 10,
        description='The maximum idle time allowed for an executor before '
        'strategy could shut down unused blocks. Default is 10 minutes.',
    )


    def _get_htex(self, label: str, num_nodes: int) -> HighThroughputExecutor:
        # If num_nodes > 0, we use srun to grab remote nodes in our allocation.
        # If num_nodes is 0, we use SimpleLauncher to run on the CURRENT node.
        if num_nodes > 0:
            launcher = WrappedLauncher(
                prepend=f'srun -l --ntasks-per-node=1 --nodes={num_nodes}',
                debug=True
            )
        else:
            launcher = SimpleLauncher()

        return HighThroughputExecutor(
            label=label,
            address='127.0.0.1', #Addressing: ValueError: 'localhost' does not appear to be an IPv4 or IPv6 address
            worker_debug=False, #enable detailed worker logging
            available_accelerators=1,
            cores_per_worker=72,
            cpu_affinity='alternating',
            prefetch_capacity=0,
            provider=LocalProvider(
                launcher=launcher,
                cmd_timeout=120,
                nodes_per_block=max(1, num_nodes), # LocalProvider needs at least 1 'block'
                init_blocks=1,
                max_blocks=1,
            ),
        )

    def get_parsl_config(self, run_dir: str | Path) -> Config:
        """Generate a Parsl configuration."""
        return Config(
            run_dir=str(run_dir),
            max_idletime=self.max_idletime,
            executors=[
                self._get_htex('train_htex', 0),
                self._get_htex('inference_htex', 0),
                self._get_htex('simulation_htex', self.num_nodes - 1),
            ],
        )


ComputeConfigTypes = Union[
    LocalConfig,
    WorkstationConfig,
    HybridWorkstationConfig,
    InferenceTrainWorkstationConfig,
    VistaConfig,
]
