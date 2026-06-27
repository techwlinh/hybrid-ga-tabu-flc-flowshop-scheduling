import numpy as np
import pytest
import os
from src.data.loader import SMTDataLoader
from src.algorithm.ga import HFGA_TS
from src.algorithm.fuzzy import FuzzyLogicController
from src.algorithm.decoder import ChromosomeDecoder
from src.config import GA_PARAMETERS, SMT_PARAMETERS

@pytest.fixture
def sample_problem():
    file_path = "refs/Antonella Branda/Problems/problem1.mat"
    return SMTDataLoader.load_mat_problem(file_path, seed=42)

def test_data_loading(sample_problem):
    """Verifies that problem dimensions and variables are correctly loaded and enriched."""
    jobs, workstations, setup_times, setup_costs = (
        sample_problem.jobs,
        sample_problem.workstations,
        sample_problem.setup_times,
        sample_problem.setup_costs
    )
    
    assert len(jobs) == 8
    assert len(workstations) == 5
    assert setup_times.shape == (8, 8, 5)
    assert setup_costs.shape == (8, 8, 5)
    
    # Check that job quantities are generated in range
    for job in jobs:
        assert 50 <= job.quantity <= 500
        assert job.priority in [1, 2, 3, 4]
        assert len(job.eligible_machines) == 5
        assert len(job.unit_processing_times) == 5

def test_batch_splitting(sample_problem):
    """Verifies that the batch splitting preserves total job quantities."""
    ga_engine = HFGA_TS(sample_problem)
    batches = ga_engine.batches
    
    # Group batches by job id
    job_quantities = {}
    for batch in batches:
        job_quantities[batch.job_id] = job_quantities.get(batch.job_id, 0) + batch.quantity
        
    # Verify that the sum of batch quantities equals the original job quantity
    for job in sample_problem.jobs:
        assert job_quantities[job.id] == job.quantity

def test_flc_bounds():
    """Verifies that FLC evaluates inputs and returns outputs within bounds."""
    flc = FuzzyLogicController()
    
    # Test multiple input combinations
    test_cases = [
        (0.0, -0.5), # Low diversity, high improvement (rapid)
        (0.5, 0.0),  # Medium diversity, stuck
        (1.0, 0.5),  # High diversity, getting worse
        (0.2, -0.2), # Random case
    ]
    
    pc_min, pc_max = GA_PARAMETERS["Crossover_Rate_Bounds"]
    pm_min, pm_max = GA_PARAMETERS["Mutation_Rate_Bounds"]
    
    for div, imp in test_cases:
        pc, pm = flc.evaluate(div, imp)
        assert pc_min <= pc <= pc_max
        assert pm_min <= pm <= pm_max

def test_decoder_hfs_constraints(sample_problem):
    """Verifies that scheduling entries satisfy the HFS and transport time constraints."""
    ga_engine = HFGA_TS(sample_problem)
    batches = ga_engine.batches
    jobs_dict = ga_engine.jobs_dict
    
    # Create a random chromosome of size 2 * N_batches
    prng = np.random.RandomState(42)
    chromosome = prng.rand(len(batches) * 2)
    
    # Decode
    res = ChromosomeDecoder.decode(
        chromosome, batches, jobs_dict,
        sample_problem.workstations, sample_problem.setup_times, sample_problem.setup_costs,
        sample_problem.transport_matrix
    )
    
    assert res.makespan > 0
    assert len(res.entries) == len(batches) * len(sample_problem.workstations)
    
    # Group entries by batch
    batch_entries = {}
    for entry in res.entries:
        batch_entries.setdefault(entry.batch_id, []).append(entry)
        
    # Check that for each batch, stage w+1 start_time is >= stage w end_time + transport_time from matrix
    for b_id, entries in batch_entries.items():
        # Sort by workstation id
        entries.sort(key=lambda x: x.workstation_id)
        
        for idx in range(len(entries) - 1):
            curr_stage = entries[idx]
            next_stage = entries[idx + 1]
            
            assert next_stage.workstation_id == curr_stage.workstation_id + 1
            # Next stage start_time must be >= current stage end_time + transport_time
            # So: next_stage.start_time - next_stage.setup_time >= curr_stage.end_time + transport_time
            setup_start = next_stage.start_time - next_stage.setup_time
            t_time = sample_problem.transport_matrix[curr_stage.workstation_id, next_stage.workstation_id]
            assert setup_start >= curr_stage.end_time + t_time - 1e-6

def test_tabu_search(sample_problem):
    """Verifies that Tabu Search optimization performs successfully without errors."""
    from src.algorithm.tabu import TabuSearch
    ga_engine = HFGA_TS(sample_problem)
    batches = ga_engine.batches
    jobs_dict = ga_engine.jobs_dict
    
    prng = np.random.RandomState(42)
    chromosome = prng.rand(len(batches) * 2)
    
    ts = TabuSearch()
    ts.max_iter = 3  # Low iteration count for speed
    
    optimized = ts.optimize(
        chromosome, batches, jobs_dict,
        sample_problem.workstations, sample_problem.setup_times, sample_problem.setup_costs,
        ChromosomeDecoder.decode, ga_engine._evaluate_fitness,
        sample_problem.transport_matrix
    )
    
    assert len(optimized) == len(chromosome)
    # The chromosome should be modified by the local search moves
    assert not np.array_equal(optimized, chromosome)
