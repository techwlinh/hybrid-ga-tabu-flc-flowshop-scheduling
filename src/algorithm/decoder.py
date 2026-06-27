import math
import numpy as np
from typing import List, Dict, Any
from src.models import Job, Batch, Workstation, Machine, ScheduleEntry, ScheduleResult
from src.config import SMT_PARAMETERS

class ChromosomeDecoder:
    """
    Decodes continuous random key chromosomes into complete, detailed schedules
    respecting all HFS constraints, quantities, setup times/costs, and transport times.
    """
    @staticmethod
    def decode(
        chromosome: np.ndarray,
        batches: List[Batch],
        jobs_dict: Dict[int, Job],
        workstations: List[Workstation],
        setup_times: np.ndarray,
        setup_costs: np.ndarray
    ) -> ScheduleResult:
        """
        Translates a chromosome of shape (2 * N_batches) into a ScheduleResult.
        """
        N_batches = len(batches)
        num_workstations = len(workstations)
        
        # 1. Split chromosome into sequence and machine genes
        seq_genes = chromosome[:N_batches]
        mach_genes = chromosome[N_batches:]
        
        # Temp list of batches to sort
        batch_routing = []
        for i, batch in enumerate(batches):
            job = jobs_dict[batch.job_id]
            batch_routing.append({
                "batch": batch,
                "priority": job.priority,
                "seq_gene": seq_genes[i],
                "mach_gene": mach_genes[i]
            })
            
        # Sort batches: primary by priority (ascending, 1 is highest), secondary by sequence gene (ascending)
        sorted_routing = sorted(batch_routing, key=lambda x: (x["priority"], x["seq_gene"]))
        
        # 2. Tracking structures for machines
        # machine_clocks[(w, m_id)] = free time of machine m_id at workstation w
        machine_clocks = {}
        # machine_last_job[(w, m_id)] = job_id of the last job processed on machine m_id at workstation w
        machine_last_job = {}
        
        for ws in workstations:
            for mach in ws.machines:
                machine_clocks[(ws.id, mach.id)] = 0.0
                machine_last_job[(ws.id, mach.id)] = None
                
        # To store start and end times of batches at each stage to enforce transport time
        # batch_completion_times[batch_id][workstation_id] = end_time
        batch_completion_times = {b.id: {} for b in batches}
        
        schedule_entries = []
        transport_time = SMT_PARAMETERS.get("Default_Transport_Time", 10.0)
        
        # 3. Schedule each batch sequentially through all workstations (stages)
        for route_info in sorted_routing:
            batch = route_info["batch"]
            mach_gene = route_info["mach_gene"]
            job = jobs_dict[batch.job_id]
            
            # Start and end time tracker for this batch across stages
            prev_stage_end = None
            
            for w in range(num_workstations):
                ws = workstations[w]
                eligible_machines = job.eligible_machines[w]
                num_eligible = len(eligible_machines)
                
                # Formula (26): Map machine gene to eligible machine index
                mach_pos = math.ceil(mach_gene * num_eligible) - 1
                mach_pos = max(0, min(mach_pos, num_eligible - 1))
                selected_machine_id = eligible_machines[mach_pos]
                
                # Ready time for the batch at workstation w
                if w == 0:
                    # Stage 0: Material arrival ready time
                    ready_time = job.material_arrival_time
                else:
                    # Stage w > 0: Completion at stage w-1 plus transport transit time
                    ready_time = prev_stage_end + transport_time
                    
                # Machine clock
                curr_mach_time = machine_clocks[(w, selected_machine_id)]
                
                # Sequence-dependent setup time and cost
                last_job_id = machine_last_job[(w, selected_machine_id)]
                if last_job_id is not None and last_job_id != job.id:
                    setup_time = float(setup_times[last_job_id, job.id, w])
                    setup_cost = float(setup_costs[last_job_id, job.id, w])
                else:
                    setup_time = 0.0
                    setup_cost = 0.0
                    
                # Schedule operations: setup and processing
                # Setup can only start when machine is free and batch has arrived
                setup_start_time = max(curr_mach_time, ready_time)
                start_time = setup_start_time + setup_time
                
                # Batch processing time at workstation w
                proc_time = batch.processing_times[w]
                end_time = start_time + proc_time
                
                # Update tracking structures
                machine_clocks[(w, selected_machine_id)] = end_time
                machine_last_job[(w, selected_machine_id)] = job.id
                batch_completion_times[batch.id][w] = end_time
                prev_stage_end = end_time
                
                # Waiting/idle time for the batch (due to queueing or setup changeovers)
                waiting_time = setup_start_time - ready_time
                
                # Save entry
                entry = ScheduleEntry(
                    batch_id=batch.id,
                    job_id=job.id,
                    workstation_id=w,
                    machine_id=selected_machine_id,
                    start_time=start_time,
                    end_time=end_time,
                    setup_time=setup_time,
                    setup_cost=setup_cost,
                    waiting_time=waiting_time
                )
                schedule_entries.append(entry)
                
        # 4. Compute overall KPIs
        if not schedule_entries:
            return ScheduleResult()
            
        makespan = max(entry.end_time for entry in schedule_entries)
        
        # Calculate Total Tardiness
        # C_j_max is the completion time of the last batch of job j at the final workstation (stage M-1)
        job_completions = {}
        for entry in schedule_entries:
            if entry.workstation_id == num_workstations - 1:
                job_id = entry.job_id
                job_completions[job_id] = max(job_completions.get(job_id, 0.0), entry.end_time)
                
        total_tardiness = 0.0
        for job_id, job in jobs_dict.items():
            comp_time = job_completions.get(job_id, 0.0)
            tardiness = max(0.0, comp_time - job.due_date)
            total_tardiness += tardiness
            
        # Setup cost and time totals
        total_setup_cost = sum(entry.setup_cost for entry in schedule_entries)
        total_setup_time = sum(entry.setup_time for entry in schedule_entries)
        
        # Machine Busy Times (Setup + Processing)
        machine_busy_time = {}
        for entry in schedule_entries:
            key = f"W{entry.workstation_id+1}_M{entry.machine_id+1}"
            duration = entry.setup_time + (entry.end_time - entry.start_time)
            machine_busy_time[key] = machine_busy_time.get(key, 0.0) + duration
            
        # Machine and Workstation Utilization rates
        machine_utilization = {}
        workstation_busy_rates = {w: [] for w in range(num_workstations)}
        
        for ws in workstations:
            for mach in ws.machines:
                key = f"W{ws.id+1}_M{mach.id+1}"
                busy = machine_busy_time.get(key, 0.0)
                rate = (busy / makespan) if makespan > 0 else 0.0
                machine_utilization[key] = min(rate, 1.0) # Cap at 100%
                workstation_busy_rates[ws.id].append(machine_utilization[key])
                
        workstation_utilization = {}
        for w_id, rates in workstation_busy_rates.items():
            workstation_utilization[w_id] = float(np.mean(rates)) if rates else 0.0
            
        return ScheduleResult(
            entries=schedule_entries,
            makespan=makespan,
            total_tardiness=total_tardiness,
            total_setup_cost=total_setup_cost,
            total_setup_time=total_setup_time,
            machine_utilization=machine_utilization,
            workstation_utilization=workstation_utilization
        )
