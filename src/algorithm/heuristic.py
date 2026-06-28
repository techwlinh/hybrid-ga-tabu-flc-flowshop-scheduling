import numpy as np
from typing import List, Dict, Tuple, Any
from src.models import Batch, Job, Workstation, Machine, ScheduleEntry, ScheduleResult, ProblemInstance
from src.config import SMT_PARAMETERS

class HeuristicScheduler:
    """
    Heuristic dispatching rules for Flow Shop scheduling:
    - FCFS (First-Come, First-Served)
    - EDD (Earliest Due Date)
    - SPT (Shortest Processing Time)
    - LPT (Longest Processing Time)
    Combined with Earliest Completion Time (ECT) machine assignment at each stage.
    """
    def __init__(self, problem: ProblemInstance, strategy: str = "EDD"):
        self.problem = problem
        self.jobs_dict = {job.id: job for job in problem.jobs}
        self.strategy = strategy.upper()
        
        # Split jobs into batches using the GA's batch splitting logic to ensure comparability
        from src.algorithm.ga import HFGA_TS
        temp_ga = HFGA_TS(problem)
        self.batches = temp_ga.batches

    def run(self, progress_callback: Any = None) -> Tuple[ScheduleResult, None, Dict[str, Any]]:
        """
        Runs the heuristic scheduling strategy.
        Returns a tuple (ScheduleResult, None, history) to match the return signature of GA/SSO.
        """
        num_workstations = len(self.problem.workstations)
        
        # 1. Sort batches based on selected strategy
        batch_info = []
        for batch in self.batches:
            job = self.jobs_dict[batch.job_id]
            batch_total_pt = float(np.sum(batch.processing_times))
            batch_info.append({
                "batch": batch,
                "job": job,
                "arrival": job.material_arrival_time,
                "due_date": job.due_date,
                "pt": batch_total_pt,
                "priority": job.priority,
                "job_id": job.id
            })
            
        if self.strategy == "FCFS":
            # Primary: Priority (1 is highest), Secondary: Arrival time, Tertiary: Job ID
            sorted_info = sorted(batch_info, key=lambda x: (x["priority"], x["arrival"], x["job_id"]))
        elif self.strategy == "EDD":
            # Primary: Priority, Secondary: Due date, Tertiary: Job ID
            sorted_info = sorted(batch_info, key=lambda x: (x["priority"], x["due_date"], x["job_id"]))
        elif self.strategy == "SPT":
            # Primary: Priority, Secondary: Total Processing Time (ascending), Tertiary: Job ID
            sorted_info = sorted(batch_info, key=lambda x: (x["priority"], x["pt"], x["job_id"]))
        elif self.strategy == "LPT":
            # Primary: Priority, Secondary: Total Processing Time (descending, so negative), Tertiary: Job ID
            sorted_info = sorted(batch_info, key=lambda x: (x["priority"], -x["pt"], x["job_id"]))
        else:
            # Fallback to EDD
            sorted_info = sorted(batch_info, key=lambda x: (x["priority"], x["due_date"], x["job_id"]))

        # 2. Tracking structures for machines
        # machine_clocks[(w, m_id)] = free time of machine m_id at workstation w
        # machine_last_job[(w, m_id)] = job_id of the last job processed on machine m_id at workstation w
        machine_clocks = {}
        machine_last_job = {}
        for ws in self.problem.workstations:
            for mach in ws.machines:
                machine_clocks[(ws.id, mach.id)] = 0.0
                machine_last_job[(ws.id, mach.id)] = None

        batch_completion_times = {b.id: {} for b in self.batches}
        schedule_entries = []

        # Transport matrix configuration
        transport_matrix = getattr(self.problem, "transport_matrix", None)
        if transport_matrix is None or getattr(transport_matrix, "shape", None) != (
            num_workstations,
            num_workstations,
        ):
            scalar_t = SMT_PARAMETERS.Default_Transport_Time
            transport_matrix = np.full((num_workstations, num_workstations), scalar_t)
            np.fill_diagonal(transport_matrix, 0.0)

        # Optional progress callback feedback
        if progress_callback is not None:
            progress_callback(generation=1, best_fitness=0.0, average_fitness=0.0, diversity=0.0, pc=0.0, pm=0.0)

        # 3. Schedule each batch sequentially through all workstation stages
        for info in sorted_info:
            batch = info["batch"]
            job = info["job"]
            prev_stage_end = None

            for w in range(num_workstations):
                ws = self.problem.workstations[w]
                eligible_machines = job.eligible_machines[w]

                # Ready time for the batch at workstation w
                if w == 0:
                    ready_time = job.material_arrival_time
                else:
                    ready_time = prev_stage_end + transport_matrix[w - 1, w]

                # Greedily find the eligible machine yielding the Earliest Completion Time (ECT)
                best_mach_id = None
                best_comp_time = float("inf")
                best_setup_time = 0.0
                best_setup_cost = 0.0

                for mach_id in eligible_machines:
                    curr_mach_time = machine_clocks[(w, mach_id)]

                    # Sequence-dependent setup time and cost
                    last_job_id = machine_last_job[(w, mach_id)]
                    if last_job_id is not None and last_job_id != job.id:
                        setup_t = float(self.problem.setup_times[last_job_id, job.id, w])
                        setup_c = float(self.problem.setup_costs[last_job_id, job.id, w])
                    else:
                        setup_t = 0.0
                        setup_c = 0.0

                    setup_start = max(curr_mach_time, ready_time)
                    start_t = setup_start + setup_t
                    proc_t = batch.processing_times[w]
                    comp_t = start_t + proc_t

                    if comp_t < best_comp_time:
                        best_comp_time = comp_t
                        best_mach_id = mach_id
                        best_setup_time = setup_t
                        best_setup_cost = setup_c

                # Schedule the batch on the selected best machine
                curr_mach_time = machine_clocks[(w, best_mach_id)]
                setup_start_time = max(curr_mach_time, ready_time)
                start_time = setup_start_time + best_setup_time
                proc_time = batch.processing_times[w]
                end_time = start_time + proc_time

                # Update clocks and tracking
                machine_clocks[(w, best_mach_id)] = end_time
                machine_last_job[(w, best_mach_id)] = job.id
                batch_completion_times[batch.id][w] = end_time
                prev_stage_end = end_time

                waiting_time = setup_start_time - ready_time

                entry = ScheduleEntry(
                    batch_id=batch.id,
                    job_id=job.id,
                    workstation_id=w,
                    machine_id=best_mach_id,
                    start_time=start_time,
                    end_time=end_time,
                    setup_time=best_setup_time,
                    setup_cost=best_setup_cost,
                    waiting_time=waiting_time,
                )
                schedule_entries.append(entry)

        # 4. Compute overall KPIs
        makespan = max(entry.end_time for entry in schedule_entries) if schedule_entries else 0.0

        # Calculate Total Tardiness
        job_completions = {}
        for entry in schedule_entries:
            if entry.workstation_id == num_workstations - 1:
                job_id = entry.job_id
                job_completions[job_id] = max(
                    job_completions.get(job_id, 0.0), entry.end_time
                )

        total_tardiness = 0.0
        for job_id, job in self.jobs_dict.items():
            comp_time = job_completions.get(job_id, 0.0)
            tardiness = max(0.0, comp_time - job.due_date)
            total_tardiness += tardiness

        # Setup cost and time totals
        total_setup_cost = sum(entry.setup_cost for entry in schedule_entries)
        total_setup_time = sum(entry.setup_time for entry in schedule_entries)

        # Machine Busy Times
        machine_busy_time = {}
        for entry in schedule_entries:
            key = f"W{entry.workstation_id+1}_M{entry.machine_id+1}"
            duration = entry.setup_time + (entry.end_time - entry.start_time)
            machine_busy_time[key] = machine_busy_time.get(key, 0.0) + duration

        # Machine and Workstation Utilization rates
        machine_utilization = {}
        workstation_busy_rates = {w: [] for w in range(num_workstations)}

        for ws in self.problem.workstations:
            for mach in ws.machines:
                key = f"W{ws.id+1}_M{mach.id+1}"
                busy = machine_busy_time.get(key, 0.0)
                rate = (busy / makespan) if makespan > 0 else 0.0
                machine_utilization[key] = min(rate, 1.0)
                workstation_busy_rates[ws.id].append(machine_utilization[key])

        workstation_utilization = {}
        for w_id, rates in workstation_busy_rates.items():
            workstation_utilization[w_id] = float(np.mean(rates)) if rates else 0.0

        # Calculate Total Tardy Units at batch level
        total_tardy_units = 0
        for batch in self.batches:
            job = self.jobs_dict[batch.job_id]
            comp_time = batch_completion_times[batch.id].get(num_workstations - 1, 0.0)
            if comp_time > job.due_date:
                total_tardy_units += batch.quantity


        res = ScheduleResult(
            entries=schedule_entries,
            makespan=makespan,
            total_tardiness=total_tardiness,
            total_setup_cost=total_setup_cost,
            total_setup_time=total_setup_time,
            machine_utilization=machine_utilization,
            workstation_utilization=workstation_utilization,
            total_tardy_units=total_tardy_units
        )


        from src.config import GA_PARAMETERS
        alpha = getattr(GA_PARAMETERS, "fitness_alpha", 0.5)
        beta = getattr(GA_PARAMETERS, "fitness_beta", 0.5)
        h_fit = float(alpha * total_tardiness + beta * makespan)

        if progress_callback is not None:
            progress_callback(generation=1, best_fitness=h_fit, average_fitness=h_fit, diversity=0.0, pc=0.0, pm=0.0)

        history = {
            "generation": [1],
            "best_fitness": [h_fit],
            "average_fitness": [h_fit],
        }

        return res, None, history
