import os
import pickle
import numpy as np
import pandas as pd
from typing import Dict, List, Any
from src.models import Workstation, Machine, Job, ProblemInstance
from src.config import SMT_PARAMETERS
from src.data.loader import SMTDataLoader


def get_saved_versions() -> List[str]:
    """Scans data_versions/ and returns sorted .pkl filenames."""
    folder = "data_versions"
    os.makedirs(folder, exist_ok=True)
    files = [f for f in os.listdir(folder) if f.endswith(".pkl")]
    return sorted(files)


def get_sorted_problem_files() -> List[str]:
    """Scans refs/Antonella Branda/Problems and returns sorted file paths."""
    folder = "refs/Antonella Branda/Problems"
    if not os.path.exists(folder):
        return []
    files = [f for f in os.listdir(folder) if f.endswith(".mat")]

    def extract_num(filename):
        digits = "".join([c for c in filename if c.isdigit()])
        return int(digits) if digits else 0

    files.sort(key=extract_num)
    return [os.path.join(folder, f) for f in files]


def sync_workstations(edited_df: pd.DataFrame, problem: Any):
    """Synchronizes the edited workstation dataframe back to the problem instance."""
    old_workstations = problem.workstations
    new_workstations = []

    num_ws = len(edited_df)

    # 1. Update Workstations and Machines
    for idx, row in enumerate(edited_df.to_dict(orient="records")):
        ws_name = str(row["Workstation Name"])
        mach_count = int(row["Machine Count"])

        ws = Workstation(id=idx, name=ws_name)
        for m in range(mach_count):
            ws.machines.append(
                Machine(id=m, workstation_id=idx, name=f"W{idx+1}_M{m+1}")
            )
        new_workstations.append(ws)

    problem.workstations = new_workstations

    # 2. Adjust setup times/costs shape: shape must be (N, N, num_ws)
    N = len(problem.jobs)
    old_num_ws = len(old_workstations)

    if num_ws != old_num_ws:
        new_setup_times = np.zeros((N, N, num_ws))
        new_setup_costs = np.zeros((N, N, num_ws))

        min_ws = min(old_num_ws, num_ws)
        new_setup_times[:, :, :min_ws] = problem.setup_times[:, :, :min_ws]
        new_setup_costs[:, :, :min_ws] = problem.setup_costs[:, :, :min_ws]

        if num_ws > old_num_ws:
            factor = SMT_PARAMETERS.get("Setup_Cost_Factor", 1.5)
            for w in range(old_num_ws, num_ws):
                rand_s = np.random.uniform(5.0, 25.0, size=(N, N))
                np.fill_diagonal(rand_s, 0.0)
                new_setup_times[:, :, w] = rand_s
                new_setup_costs[:, :, w] = rand_s * factor

        problem.setup_times = new_setup_times
        problem.setup_costs = new_setup_costs

        # Adjust transport matrix
        new_transport = np.zeros((num_ws, num_ws))
        if getattr(problem, "transport_matrix", None) is not None:
            min_t_ws = min(len(problem.transport_matrix), num_ws)
            new_transport[:min_t_ws, :min_t_ws] = problem.transport_matrix[
                :min_t_ws, :min_t_ws
            ]
        else:
            min_t_ws = 0

        if num_ws > old_num_ws:
            for i in range(num_ws):
                for j in range(num_ws):
                    if i >= old_num_ws or j >= old_num_ws:
                        if i == j:
                            new_transport[i, j] = 0.0
                        else:
                            new_transport[i, j] = round(np.random.uniform(5.0, 15.0), 1)
        problem.transport_matrix = new_transport

    # 3. Synchronize Jobs' eligible_machines and unit_processing_times length
    for job in problem.jobs:
        old_proc = job.unit_processing_times
        new_proc = np.zeros(num_ws)
        min_p_ws = min(len(old_proc), num_ws)
        new_proc[:min_p_ws] = old_proc[:min_p_ws]

        if num_ws > old_num_ws:
            new_proc[old_num_ws:] = np.random.uniform(
                50.0, 150.0, size=num_ws - old_num_ws
            )
        job.unit_processing_times = np.round(new_proc, 2)

        new_eligible = {}
        for w in range(num_ws):
            if w < old_num_ws and w in job.eligible_machines:
                old_elig = job.eligible_machines[w]
                mach_limit = len(problem.workstations[w].machines)
                valid_elig = [m for m in old_elig if m < mach_limit]
                if not valid_elig:
                    valid_elig = [0]
                new_eligible[w] = valid_elig
            else:
                new_eligible[w] = list(range(len(problem.workstations[w].machines)))
        job.eligible_machines = new_eligible


def sync_jobs(edited_df: pd.DataFrame, problem: Any, mach_tuples: list):
    """Synchronizes the edited jobs dataframe back to the problem instance."""
    for idx, row in edited_df.iterrows():
        job_id = int(row["Job ID"])
        job = next((j for j in problem.jobs if j.id == job_id), None)
        if job is None:
            continue

        job.quantity = int(row["Quantity (units)"])
        job.priority = int(row["Priority Group"])
        job.material_arrival_time = float(row["Material Arrival (min)"])
        job.due_date = float(row["Due Date (min)"])

        # Reconstruct eligible_machines dictionary
        eligible = {ws.id: [] for ws in problem.workstations}
        for ws_id, mach_id, col_name in mach_tuples:
            if ws_id in eligible and row.get(col_name, False):
                eligible[ws_id].append(mach_id)

        # Ensure at least 1 machine is eligible for each workstation stage
        for ws_id in eligible:
            if not eligible[ws_id] and len(problem.workstations[ws_id].machines) > 0:
                eligible[ws_id] = [0]  # default to machine 0

        job.eligible_machines = eligible


def explain_solution(
    result: Any, jobs: List[Any], workstations: List[Any], batches: List[Any]
) -> Dict[str, Any]:
    """Generates explainable natural language descriptions of the scheduling decisions."""
    # 1. Bottleneck identification
    ws_util = result.workstation_utilization
    sorted_ws = sorted(ws_util.items(), key=lambda x: x[1], reverse=True)
    bottleneck_ws_id = sorted_ws[0][0]
    bottleneck_ws_name = workstations[bottleneck_ws_id].name

    # 2. Critical tardy jobs
    tardy_jobs = []
    job_completions = {}
    num_ws = len(workstations)
    for entry in result.entries:
        if entry.workstation_id == num_ws - 1:
            job_completions[entry.job_id] = max(
                job_completions.get(entry.job_id, 0.0), entry.end_time
            )

    for job in jobs:
        comp_time = job_completions.get(job.id, 0.0)
        tardiness = max(0.0, comp_time - job.due_date)
        if tardiness > 0.0:
            # Analyze reason
            reason = "Late scheduled"
            if job.material_arrival_time > job.due_date * 0.8:
                reason = f"Late material arrival time ({job.material_arrival_time:.1f} min relative to due date {job.due_date:.1f} min)"
            elif job.priority == 3:
                reason = "Low priority level (Group 3), pushed back behind priority Groups 1 & 2"
            else:
                # Calculate changeover overhead
                job_setup_entries = [
                    e for e in result.entries if e.job_id == job.id and e.setup_time > 0
                ]
                total_job_setup = sum(e.setup_time for e in job_setup_entries)
                if total_job_setup > 20:
                    reason = f"High changeover/setup time overhead ({total_job_setup:.1f} min spent in setups across lines)"
                else:
                    reason = "High volume processing bottleneck at workstations"

            tardy_jobs.append(
                {
                    "id": job.id,
                    "due_date": job.due_date,
                    "completion": comp_time,
                    "tardiness": tardiness,
                    "priority": job.priority,
                    "reason": reason,
                }
            )

    # 3. Batch splitting summary
    splits_count = 0
    job_batches = {}
    for b in batches:
        job_batches[b.job_id] = job_batches.get(b.job_id, 0) + 1

    split_details = []
    for j_id, count in job_batches.items():
        if count > 1:
            splits_count += 1
            split_details.append(
                f"Job {j_id} split into {count} batches to distribute its quantity of {jobs[j_id].quantity} units"
            )

    return {
        "bottleneck_id": bottleneck_ws_id,
        "bottleneck_name": bottleneck_ws_name,
        "bottleneck_util": ws_util[bottleneck_ws_id] * 100,
        "tardy_jobs": tardy_jobs,
        "splits_count": splits_count,
        "split_details": split_details[:5],  # Top 5 details
    }
