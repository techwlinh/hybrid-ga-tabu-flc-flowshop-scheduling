from dataclasses import dataclass, field
from typing import List, Dict, Any
import numpy as np


@dataclass
class Job:
    """Represents a production job order containing PCB assembly details."""

    id: int
    quantity: int
    due_date: float
    priority: int  # Priority level (e.g. 1, 2, 3 where 1 is highest priority)
    material_arrival_time: float
    unit_processing_times: (
        np.ndarray
    )  # Array of shape (M,) defining unit processing time at each workstation stage
    # Dictionary mapping workstation_id -> list of eligible machine_ids at that workstation
    eligible_machines: Dict[int, List[int]] = field(default_factory=dict)


@dataclass
class Batch:
    """Represents a sub-lot (batch) split from a parent job for parallel flow-shop balance."""

    id: str  # Unique batch ID, e.g. "J0_B0", "J0_B1"
    job_id: int
    batch_index: int
    quantity: int
    processing_times: (
        np.ndarray
    )  # Array of shape (M,) scaled by batch quantity (unit_processing_times * quantity)


@dataclass
class Machine:
    """Represents a single physical machine (e.g. SMT placing nozzle) on a workstation line."""

    id: int
    workstation_id: int
    name: str


@dataclass
class Workstation:
    """Represents a manufacturing stage (e.g. Printer, SPI, Mounter) containing parallel machines."""

    id: int
    name: str
    machines: List[Machine] = field(default_factory=list)


@dataclass
class ScheduleEntry:
    """Represents a single scheduled operation of a batch on a specific machine at a workstation."""

    batch_id: str
    job_id: int
    workstation_id: int
    machine_id: int
    start_time: float
    end_time: float
    setup_time: float
    setup_cost: float
    waiting_time: (
        float  # Waiting time (delay from material arrival or transport transit)
    )


@dataclass
class ScheduleResult:
    """Contains full scheduling solution metrics, KPIs, and dispatch lists."""

    entries: List[ScheduleEntry] = field(default_factory=list)
    makespan: float = 0.0
    total_tardiness: float = 0.0
    total_setup_cost: float = 0.0
    total_setup_time: float = 0.0
    # Dictionary mapping unique machine ID (e.g. "W2_M1") to its utilization rate (busy_time / makespan)
    machine_utilization: Dict[str, float] = field(default_factory=dict)
    # Dictionary mapping workstation ID to average utilization rate of its machines
    workstation_utilization: Dict[int, float] = field(default_factory=dict)


@dataclass
class ProblemInstance:
    """Encapsulates all jobs, workstations, setup, and cost parameters of a problem instance."""

    jobs: List[Job]
    workstations: List[Workstation]
    setup_times: np.ndarray
    setup_costs: np.ndarray
    transport_matrix: np.ndarray = None
