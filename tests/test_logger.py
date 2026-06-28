import os
import shutil
import json
import pandas as pd
import pytest
from src.utils.logger import save_experiment_data, get_next_experiment_number
from src.models import ScheduleResult, ScheduleEntry

@pytest.fixture
def temp_experiments_dir():
    dir_path = "temp_experiments_test"
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
    yield dir_path
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)

def test_experiment_numbering(temp_experiments_dir):
    # Empty dir should start at 1
    assert get_next_experiment_number(temp_experiments_dir) == 1
    
    # Create some dummy folders
    os.makedirs(os.path.join(temp_experiments_dir, "001_20260628_120000_Problem_1"))
    assert get_next_experiment_number(temp_experiments_dir) == 2
    
    os.makedirs(os.path.join(temp_experiments_dir, "005_20260628_120000_Problem_1"))
    assert get_next_experiment_number(temp_experiments_dir) == 6

def test_save_experiment_data(temp_experiments_dir):
    dataset_name = "Problem_Test"
    params = {"pop_size": 10, "max_gens": 5}
    
    # Construct dummy schedule result
    dummy_res = ScheduleResult(
        entries=[],
        makespan=150.0,
        total_tardiness=10.0,
        total_setup_cost=50.0,
        total_setup_time=5.0
    )
    
    run_results = {
        "selected_algs": ["Heuristic (EDD)", "HFGA-TS"],
        "num_runs": 2,
        "algs": {
            "Heuristic (EDD)": {
                "best_result": dummy_res,
                "best_chrom": None,
                "best_history": {
                    "generation": [1],
                    "best_fitness": [10.0],
                    "average_fitness": [10.0]
                },
                "runs": [
                    {"makespan": 150.0, "total_tardiness": 10.0, "total_setup_cost": 50.0, "total_setup_time": 5.0, "duration": 0.5},
                    {"makespan": 150.0, "total_tardiness": 10.0, "total_setup_cost": 50.0, "total_setup_time": 5.0, "duration": 0.5}
                ]
            },
            "HFGA-TS": {
                "best_result": dummy_res,
                "best_chrom": None,
                "best_history": {
                    "generation": [1, 2, 3],
                    "best_fitness": [15.0, 12.0, 10.0],
                    "average_fitness": [20.0, 18.0, 15.0]
                },
                "runs": [
                    {"makespan": 150.0, "total_tardiness": 10.0, "total_setup_cost": 50.0, "total_setup_time": 5.0, "duration": 1.2},
                    {"makespan": 160.0, "total_tardiness": 12.0, "total_setup_cost": 60.0, "total_setup_time": 6.0, "duration": 1.1}
                ]
            }
        }
    }
    
    saved_folder = save_experiment_data(dataset_name, params, run_results, temp_experiments_dir)
    
    # Check that folder exists
    assert os.path.exists(saved_folder)
    
    # Check file creations
    files = os.listdir(saved_folder)
    assert len(files) == 3
    
    csv_file = [f for f in files if f.endswith(".csv")][0]
    json_file = [f for f in files if f.endswith(".json")][0]
    png_file = [f for f in files if f.endswith(".png")][0]
    
    # Read CSV
    df = pd.read_csv(os.path.join(saved_folder, csv_file))
    assert list(df.columns) == ["Generation", "Heuristic (EDD)", "HFGA-TS"]
    assert len(df) == 3 # HFGA-TS has max 3 generations
    # Check carry-forward of Heuristic (EDD)
    assert list(df["Heuristic (EDD)"]) == [10.0, 10.0, 10.0]
    
    # Read JSON
    with open(os.path.join(saved_folder, json_file), "r", encoding="utf-8") as f:
        meta = json.load(f)
        
    assert meta["dataset_name"] == dataset_name
    assert meta["number_of_runs"] == 2
    assert "Heuristic (EDD)" in meta["results"]
    assert "HFGA-TS" in meta["results"]
