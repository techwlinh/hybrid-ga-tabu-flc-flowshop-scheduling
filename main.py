import argparse
import os
import time
import numpy as np
import pandas as pd

from src.data.loader import SMTDataLoader
from src.algorithm.ga import HFGA_TS
from src.visualization.gantt import SMTGanttChart
from src.config import GA_PARAMETERS, SMT_PARAMETERS


def main():
    parser = argparse.ArgumentParser(
        description="HFGA-TS SMT Scheduling System CLI Runner"
    )
    parser.add_argument(
        "--file",
        type=str,
        default="refs/Antonella Branda/Problems/problem1.mat",
        help="Path to the MATLAB problem instance .mat file",
    )
    parser.add_argument("--pop", type=int, default=100, help="GA Population Size")
    parser.add_argument(
        "--gen",
        type=int,
        default=100,
        help="GA Maximum Generations (default 100 for fast CLI)",
    )
    parser.add_argument(
        "--split", type=float, default=150.0, help="Batch Splitting Threshold T"
    )
    parser.add_argument(
        "--transport",
        type=float,
        default=10.0,
        help="Transit time between workstations (min)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for enrichment"
    )
    parser.add_argument(
        "--sequential",
        action="store_true",
        help="Disable multiprocessing and run sequentially",
    )

    args = parser.parse_args()

    print("======================================================================")
    print("      🚀 SMT FLOW SHOP SCHEDULING OPTIMIZER RUNNER (HFGA-TS)")
    print("======================================================================")
    print(f"Loading Problem File: {args.file}")

    # Configure parameters from arguments
    GA_PARAMETERS.Population_Size = args.pop
    GA_PARAMETERS.Maximum_Generation = args.gen
    GA_PARAMETERS.Threshold_of_Batch_Splitting = args.split
    SMT_PARAMETERS.Default_Transport_Time = args.transport
    GA_PARAMETERS.Use_Parallel_Execution = not args.sequential

    # 1. Load Problem
    try:
        problem = SMTDataLoader.load_mat_problem(args.file, seed=args.seed)
        # Override the random transport matrix with the constant transport time from CLI argument
        num_ws = len(problem.workstations)
        problem.transport_matrix = np.full((num_ws, num_ws), args.transport)
        np.fill_diagonal(problem.transport_matrix, 0.0)
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return

    print(f"✔ Problem Loaded Successfully.")
    print(f"  • Total Jobs: {len(problem.jobs)}")
    print(f"  • Total Workstations: {len(problem.workstations)}")
    total_m = sum(len(ws.machines) for ws in problem.workstations)
    print(f"  • Total SMT Lines (Parallel Machines): {total_m}")

    # 2. Initialize GA Optimization Engine
    print("\nInitializing Genetic Algorithm Engine...")
    ga_engine = HFGA_TS(problem)
    print(
        f"  • Preprocessing split: {len(ga_engine.batches)} total batches generated from {len(problem.jobs)} jobs."
    )

    print("\nRunning GA Optimization Loop...")
    t_start = time.time()

    # Simple console logger callback
    def progress_log(generation, best_fitness, average_fitness, diversity, pc, pm):
        if generation % 10 == 0 or generation == 1 or generation == args.gen:
            print(
                f"  [Gen {generation:4d}/{args.gen}] Best Fit (Tardiness): {best_fitness:8.1f} min | Avg: {average_fitness:8.1f} min | Div: {diversity:5.3f} | Pc: {pc:.3f} | Pm: {pm:.3f}"
            )

    res, best_chrom, history = ga_engine.run(progress_callback=progress_log)
    duration = time.time() - t_start

    print("✔ Optimization Complete!")
    print(f"  • Execution Time: {duration:.2f} seconds")

    # 3. Output KPI Summary
    print("\n======================================================================")
    print("      📊 KEY PERFORMANCE INDICATORS (KPI) SUMMARY")
    print("======================================================================")
    print(f"  • Makespan (Total Line Span):      {res.makespan:.1f} minutes")
    print(f"  • Total Tardiness (Job Delays):     {res.total_tardiness:.1f} minutes")
    print(f"  • Total Changeover Setup Cost:     ${res.total_setup_cost:.1f}")
    print(f"  • Total Changeover Setup Time:     {res.total_setup_time:.1f} minutes")

    avg_util = np.mean(list(res.machine_utilization.values())) * 100
    print(f"  • Average Machine Utilization:     {avg_util:.1f}%")

    # Workstation level utilization breakdown
    print("\n  Workstation Utilization Breakdown:")
    for w_id, rate in res.workstation_utilization.items():
        print(
            f"    - Workstation {w_id+1} ({problem.workstations[w_id].name}): {rate*100:5.1f}%"
        )

    # 4. Save results to disk
    # Save schedule entries log
    log_file = "smt_schedule_log.csv"
    entries_data = []
    for entry in res.entries:
        entries_data.append(
            {
                "Batch_ID": entry.batch_id,
                "Job_ID": entry.job_id,
                "Workstation_ID": entry.workstation_id,
                "Machine_ID": entry.machine_id,
                "Start_Time": entry.start_time,
                "End_Time": entry.end_time,
                "Setup_Time": entry.setup_time,
                "Setup_Cost": entry.setup_cost,
                "Waiting_Time": entry.waiting_time,
            }
        )
    df_sched = pd.DataFrame(entries_data)
    df_sched.to_csv(log_file, index=False)
    print(f"\n💾 Saved schedule entries log to: {os.path.abspath(log_file)}")

    # Save static Gantt chart
    gantt_file = "smt_schedule_gantt.png"
    SMTGanttChart.plot_static_gantt(
        res, ga_engine.jobs_dict, problem.workstations, output_path=gantt_file
    )
    print(f"🖼 Saved schedule Gantt chart image to: {os.path.abspath(gantt_file)}")

    # 5. Simple Explainable Bottleneck Analysis
    sorted_ws = sorted(
        res.workstation_utilization.items(), key=lambda x: x[1], reverse=True
    )
    b_id, b_rate = sorted_ws[0]
    print("\n======================================================================")
    print("      🧠 EXPLAINABLE OPTIMIZATION HIGHLIGHTS")
    print("======================================================================")
    print(
        f"  • Bottleneck Stage: Workstation {b_id+1} ({problem.workstations[b_id].name}) has the highest utilization rate at {b_rate*100:.1f}%."
    )
    print("    Downstream operations are dependent on scheduling gates here.")

    # Count tardy jobs
    num_ws = len(problem.workstations)
    job_completions = {}
    for entry in res.entries:
        if entry.workstation_id == num_ws - 1:
            job_completions[entry.job_id] = max(
                job_completions.get(entry.job_id, 0.0), entry.end_time
            )

    tardy_jobs_count = 0
    for j_id, job in ga_engine.jobs_dict.items():
        if job_completions.get(j_id, 0.0) > job.due_date:
            tardy_jobs_count += 1

    if tardy_jobs_count == 0:
        print(
            "  • Schedule Quality: Excellent! All job order due dates have been fully met."
        )
    else:
        print(
            f"  • Schedule Quality: Good. {tardy_jobs_count} job orders experienced tardiness due to material/capacity constraints."
        )
        print(
            "    Refer to the Streamlit app's 'Solution Explainer' tab for a detailed breakdown of each tardy job."
        )
    print("======================================================================\n")


if __name__ == "__main__":
    main()
