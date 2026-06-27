import numpy as np
import pytest
import pandas as pd
import plotly.graph_objects as go
from src.data.loader import SMTDataLoader
from src.algorithm.ga import HFGA_TS
from src.algorithm.decoder import ChromosomeDecoder
from src.visualization.stats import (
    get_scheduling_stats_and_charts,
    plot_machine_workload,
    plot_job_performance_scatter,
    plot_job_status_donut,
    plot_workstation_utilization,
)

@pytest.fixture
def sample_problem():
    file_path = "refs/Antonella Branda/Problems/problem1.mat"
    return SMTDataLoader.load_mat_problem(file_path, seed=42)

def test_stats_computation(sample_problem):
    """Verifies that scheduling performance statistics and metrics are computed correctly."""
    ga_engine = HFGA_TS(sample_problem)
    batches = ga_engine.batches
    jobs_dict = ga_engine.jobs_dict
    jobs = sample_problem.jobs
    workstations = sample_problem.workstations
    
    # Generate a random chromosome and decode it to get a ScheduleResult
    prng = np.random.RandomState(42)
    chromosome = prng.rand(len(batches) * 2)
    
    res = ChromosomeDecoder.decode(
        chromosome, batches, jobs_dict,
        workstations, sample_problem.setup_times, sample_problem.setup_costs,
        sample_problem.transport_matrix
    )
    
    # Run stats calculation
    stats = get_scheduling_stats_and_charts(res, jobs, workstations)
    
    # 1. Assert overall metrics are within expected ranges
    assert isinstance(stats["lbe_overall"], float)
    assert 0.0 <= stats["lbe_overall"] <= 100.0
    
    assert isinstance(stats["si_overall"], float)
    assert stats["si_overall"] >= 0.0
    
    assert isinstance(stats["tardy_rate"], float)
    assert 0.0 <= stats["tardy_rate"] <= 100.0
    
    assert stats["total_jobs"] == len(jobs)
    assert 0 <= stats["tardy_jobs_count"] <= len(jobs)
    
    # 2. Check DataFrames structures
    df_machines = stats["df_machines"]
    assert isinstance(df_machines, pd.DataFrame)
    assert not df_machines.empty
    assert set(["Machine", "Workstation", "Workstation ID", "Time (min)", "Activity"]).issubset(df_machines.columns)
    
    df_jobs = stats["df_jobs"]
    assert isinstance(df_jobs, pd.DataFrame)
    assert not df_jobs.empty
    assert set(["Job ID", "Job Number", "Priority", "Priority Int", "Due Date (min)", "Completion Time (min)", "Tardiness (min)", "Status"]).issubset(df_jobs.columns)

    # 3. Check workstation utilization
    ws_utilization = stats["ws_utilization"]
    assert isinstance(ws_utilization, dict)
    assert len(ws_utilization) == len(workstations)
    for ws_id, util in ws_utilization.items():
        assert 0.0 <= util <= 100.0

def test_stats_plots(sample_problem):
    """Verifies that all Plotly functions successfully return Plotly Figure objects."""
    ga_engine = HFGA_TS(sample_problem)
    batches = ga_engine.batches
    jobs_dict = ga_engine.jobs_dict
    jobs = sample_problem.jobs
    workstations = sample_problem.workstations
    
    prng = np.random.RandomState(42)
    chromosome = prng.rand(len(batches) * 2)
    
    res = ChromosomeDecoder.decode(
        chromosome, batches, jobs_dict,
        workstations, sample_problem.setup_times, sample_problem.setup_costs,
        sample_problem.transport_matrix
    )
    
    stats = get_scheduling_stats_and_charts(res, jobs, workstations)
    
    # Render Plotly charts
    fig_workload = plot_machine_workload(stats["df_machines"], "All")
    assert isinstance(fig_workload, go.Figure)
    
    fig_workload_filtered = plot_machine_workload(stats["df_machines"], workstations[0].name)
    assert isinstance(fig_workload_filtered, go.Figure)
    
    fig_scatter = plot_job_performance_scatter(stats["df_jobs"])
    assert isinstance(fig_scatter, go.Figure)
    
    fig_donut = plot_job_status_donut(stats["df_jobs"])
    assert isinstance(fig_donut, go.Figure)
    
    fig_ws_util = plot_workstation_utilization(stats["ws_utilization"], workstations)
    assert isinstance(fig_ws_util, go.Figure)

def test_empty_schedule_handling():
    """Verifies that stats helper handles empty/none schedules gracefully."""
    stats = get_scheduling_stats_and_charts(None, [], [])
    assert stats["lbe_overall"] == 0.0
    assert stats["si_overall"] == 0.0
    assert stats["tardy_rate"] == 0.0
    assert stats["df_machines"].empty
    assert stats["df_jobs"].empty
