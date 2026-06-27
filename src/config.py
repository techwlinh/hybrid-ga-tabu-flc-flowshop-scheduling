# Global Configuration and Hyperparameters for HFGA-TS SMT Scheduling System

# 1. Genetic Algorithm (GA) Hyperparameters
GA_PARAMETERS = {
    "Threshold_of_Batch_Splitting": 150.0,    # Threshold T to split large jobs
    "Maximum_Generation": 1000,               # Maximum number of GA generations
    "Population_Size": 100,                   # Number of chromosomes in population
    "No_Improvement_Limit": 100,              # Stall limit before triggering Tabu Search
    "Elitism_Rate": 0.1,                      # Portion of top solutions copied directly
    "Initial_Crossover_Rate": 0.8,            # Initial crossover probability (P_c)
    "Initial_Mutation_Rate": 0.1,             # Initial mutation probability (P_m)
    "Crossover_Rate_Bounds": [0.5, 0.95],     # Bounds for FLC crossover rate adjustment
    "Mutation_Rate_Bounds": [0.05, 0.2],      # Bounds for FLC mutation rate adjustment
    "Max_Iterations_of_Tabu_Search": 30,      # Max iterations per Tabu Search run
    "Tabu_List_Size": 15,                     # Length of the tabu memory list
    "Time_Limitation_Seconds": 600,           # Global runtime cap (10 minutes)
    "Use_Parallel_Execution": True,           # Enable multiprocessing parallel evaluations
}

# 2. SMT Flow Shop Environment Parameters
SMT_PARAMETERS = {
    # Default number of identical parallel SMT machines at each Workstation stage
    "Default_Machines_Per_Workstation": {
        0: 2,  # Stage 0: e.g., Printing
        1: 2,  # Stage 1: e.g., SPI
        2: 3,  # Stage 2: e.g., High-speed placement
        3: 2,  # Stage 3: e.g., Multi-functional placement
        4: 2,  # Stage 4: e.g., Reflow
    },
    # Transport/transit duration (time units) required to move a batch from stage w to w+1
    "Default_Transport_Time": 10.0,
    
    # Multiplying factor: setup_cost = setup_time * Setup_Cost_Factor
    "Setup_Cost_Factor": 1.5,
    
    # Eligible machines selection density (portion of machines in a workstation a job can run on)
    "Machine_Eligibility_Density": 0.8,
}
