import pytest
import numpy as np
from src.data.loader import SMTDataLoader
from src.algorithm.ga import HFGA_TS
from src.algorithm.heuristic import HeuristicScheduler
from src.algorithm.sso import SSOScheduler

@pytest.fixture
def sample_problem():
    file_path = "data_versions/raw/problem1.mat"
    return SMTDataLoader.load_mat_problem(file_path, seed=42)

def test_heuristic_strategies(sample_problem):
    """Verifies that all 4 heuristic strategies execute successfully and return valid ScheduleResults."""
    strategies = ["FCFS", "EDD", "SPT", "LPT"]
    
    for strat in strategies:
        scheduler = HeuristicScheduler(sample_problem, strategy=strat)
        res, chrom, hist = scheduler.run()
        
        # Verify result metrics
        assert res.makespan > 0
        assert res.total_tardiness >= 0
        assert res.total_setup_cost >= 0
        assert len(res.entries) == len(scheduler.batches) * len(sample_problem.workstations)
        
        # Verify chronological ordering for each batch stage
        batch_entries = {}
        for entry in res.entries:
            batch_entries.setdefault(entry.batch_id, []).append(entry)
            
        for b_id, entries in batch_entries.items():
            entries.sort(key=lambda x: x.workstation_id)
            for idx in range(len(entries) - 1):
                curr_stage = entries[idx]
                next_stage = entries[idx+1]
                
                assert next_stage.workstation_id == curr_stage.workstation_id + 1
                setup_start = next_stage.start_time - next_stage.setup_time
                t_time = sample_problem.transport_matrix[curr_stage.workstation_id, next_stage.workstation_id]
                assert setup_start >= curr_stage.end_time + t_time - 1e-6

def test_sso_execution(sample_problem):
    """Verifies that SSO swarm runs and returns a valid ScheduleResult."""
    sso = SSOScheduler(sample_problem)
    sso.max_generations = 3  # Low generations for fast execution
    sso.pop_size = 10
    
    res, best_pos, hist = sso.run()
    
    assert res.makespan > 0
    assert len(best_pos) == sso.N_batches * 2
    assert len(hist["generation"]) == 3

def test_ga_parameterizations(sample_problem):
    """Verifies that the modified HFGA_TS can execute with FLC and TS toggled off."""
    # Test Standard GA (no FLC, no TS)
    ga_std = HFGA_TS(sample_problem, use_flc=False, use_tabu=False)
    ga_std.max_generations = 3
    ga_std.pop_size = 10
    res_std, _, hist_std = ga_std.run()
    assert res_std.makespan > 0
    
    # Test FLC + GA (no TS)
    ga_flc = HFGA_TS(sample_problem, use_flc=True, use_tabu=False)
    ga_flc.max_generations = 3
    ga_flc.pop_size = 10
    res_flc, _, hist_flc = ga_flc.run()
    assert res_flc.makespan > 0
