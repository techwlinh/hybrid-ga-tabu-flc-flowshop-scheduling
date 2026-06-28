# Global Configuration and Hyperparameters for HFGA-TS SMT Scheduling System

from pydantic import BaseModel, Field
from typing import Dict, List


class GAParameters(BaseModel):
    Threshold_of_Batch_Splitting: float = Field(
        default=150.0, description="Threshold T to split large jobs"
    )
    Maximum_Generation: int = Field(
        default=1000, description="Maximum number of GA generations"
    )
    Population_Size: int = Field(
        default=100, description="Number of chromosomes in population"
    )
    No_Improvement_Limit: int = Field(
        default=100, description="Stall limit before triggering Tabu Search"
    )
    Elitism_Rate: float = Field(
        default=0.1, description="Portion of top solutions copied directly"
    )
    Initial_Crossover_Rate: float = Field(
        default=0.8, description="Initial crossover probability (P_c)"
    )
    Initial_Mutation_Rate: float = Field(
        default=0.1, description="Initial mutation probability (P_m)"
    )
    Crossover_Rate_Bounds: List[float] = Field(
        default_factory=lambda: [0.5, 0.95],
        description="Bounds for FLC crossover rate adjustment",
    )
    Mutation_Rate_Bounds: List[float] = Field(
        default_factory=lambda: [0.05, 0.2],
        description="Bounds for FLC mutation rate adjustment",
    )
    Max_Iterations_of_Tabu_Search: int = Field(
        default=30, description="Max iterations per Tabu Search run"
    )
    Tabu_List_Size: int = Field(
        default=15, description="Length of the tabu memory list"
    )
    Time_Limitation_Seconds: int = Field(
        default=600, description="Global runtime cap (10 minutes)"
    )
    Use_Parallel_Execution: bool = Field(
        default=True, description="Enable multiprocessing parallel evaluations"
    )
    fitness_alpha: float = Field(
        default=0.8, description="Weight alpha for total tardiness in fitness"
    )
    fitness_beta: float = Field(
        default=0.2, description="Weight beta for makespan in fitness"
    )


class SMTParameters(BaseModel):
    # Default number of identical parallel SMT machines at each Workstation stage
    Default_Machines_Per_Workstation: Dict[int, int] = Field(
        default_factory=lambda: {
            0: 4,  # Stage 0: e.g., Printing
            1: 5,  # Stage 1: e.g., SPI
            2: 3,  # Stage 2: e.g., High-speed placement
            3: 4,  # Stage 3: e.g., Multi-functional placement
            4: 3,  # Stage 4: e.g., Reflow
        },
        description="Default number of identical parallel SMT machines at each Workstation stage",
    )
    # Transport/transit duration (time units) required to move a batch from stage w to w+1
    Default_Transport_Time: float = Field(
        default=10.0, description="Transport/transit duration"
    )
    # Multiplying factor: setup_cost = setup_time * Setup_Cost_Factor
    Setup_Cost_Factor: float = Field(default=1.5, description="Setup cost factor")
    # Eligible machines selection density (portion of machines in a workstation a job can run on)
    Machine_Eligibility_Density: float = Field(
        default=0.8, description="Eligible machines selection density"
    )


# Instantiate configuration objects
GA_PARAMETERS = GAParameters()
SMT_PARAMETERS = SMTParameters()
