import time
import numpy as np
import streamlit as st
from typing import Any

from src.algorithm.ga import HFGA_TS
from src.algorithm.heuristic import HeuristicScheduler
from src.algorithm.sso import SSOScheduler
from src.utils.logger import save_experiment_data


def render_optimization_tab(
    problem_instance: Any,
    max_gens: int,
    dataset_name: str = "unknown",
    params: dict = None,
):
    """Renders Tab 2: Optimization Run with comparative algorithm selection."""
    st.subheader("Run Comparative Scheduling Experiments")

    st.write("### 🛠️ Select Algorithms to Test")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Heuristic Baselines (Deterministic)**")
        run_fcfs = st.checkbox(
            "Heuristic (FCFS)",
            value=True,
            help="First-Come First-Served sequencing with Earliest Completion Time (ECT) machine assignment.",
        )
        run_edd = st.checkbox(
            "Heuristic (EDD)",
            value=True,
            help="Earliest Due Date sequencing with ECT machine assignment.",
        )
        run_spt = st.checkbox(
            "Heuristic (SPT)",
            value=False,
            help="Shortest Processing Time sequencing with ECT machine assignment.",
        )
        run_lpt = st.checkbox(
            "Heuristic (LPT)",
            value=False,
            help="Longest Processing Time sequencing with ECT machine assignment.",
        )
    with col2:
        st.markdown("**Metaheuristic Optimizers (Stochastic)**")
        run_ga = st.checkbox(
            "Standard GA (no FLC, no TS)",
            value=False,
            help="Genetic Algorithm with static crossover and mutation rates.",
        )
        run_ga_tabu = st.checkbox(
            "Hybrid GA-Tabu (no FLC)",
            value=False,
            help="Genetic Algorithm with static crossover/mutation, triggering Tabu Search on stalls.",
        )
        run_flc_ga = st.checkbox(
            "FLC + GA (no TS)",
            value=False,
            help="Fuzzy Logic Controller GA with adaptive rates, without Tabu Search.",
        )
        run_sso = st.checkbox(
            "SSO (Simplified Swarm Optimization)",
            value=True,
            help="SSO swarm search algorithm sharing same chromosome representation.",
        )
        run_hfgats = st.checkbox(
            "Proposed HFGA-TS (Full Hybrid)",
            value=True,
            help="Our full hybrid fuzzy GA with Tabu Search local search.",
        )

    st.write("### ⚙️ Statistical Parameters")
    num_runs = st.slider(
        "Number of Runs (R) for Stochastic Algorithms",
        min_value=1,
        max_value=10,
        value=5,
        help="Run metaheuristics multiple times with different seeds to calculate mean and standard deviation.",
    )

    # Optimization control button
    start_opt = st.button("🚀 Start Comparative Optimization")

    # Training progress placeholders
    progress_bar = st.progress(0.0)
    status_text = st.empty()
    realtime_metric_container = st.empty()
    realtime_chart_container = st.empty()

    if start_opt:
        selected_algs = []
        if run_fcfs:
            selected_algs.append("Heuristic (FCFS)")
        if run_edd:
            selected_algs.append("Heuristic (EDD)")
        if run_spt:
            selected_algs.append("Heuristic (SPT)")
        if run_lpt:
            selected_algs.append("Heuristic (LPT)")
        if run_ga:
            selected_algs.append("Standard GA")
        if run_ga_tabu:
            selected_algs.append("GA-Tabu")
        if run_flc_ga:
            selected_algs.append("FLC-GA")
        if run_sso:
            selected_algs.append("SSO")
        if run_hfgats:
            selected_algs.append("HFGA-TS")

        if not selected_algs:
            st.error("Please select at least one algorithm to run.")
            return

        st.session_state["run_results"] = {
            "selected_algs": selected_algs,
            "num_runs": num_runs,
            "algs": {},
        }

        total_tasks = len(selected_algs)
        task_idx = 0

        # Setup common batches and jobs_dict reference for visualization
        temp_ga = HFGA_TS(problem_instance)
        st.session_state["batches"] = temp_ga.batches
        st.session_state["jobs_dict"] = temp_ga.jobs_dict

        from src.config import GA_PARAMETERS
        alpha = getattr(GA_PARAMETERS, "fitness_alpha", 0.5)
        beta = getattr(GA_PARAMETERS, "fitness_beta", 0.5)

        for alg_name in selected_algs:
            status_text.markdown(f"🏃 Running **{alg_name}**...")
            runs_data = []
            best_res = None
            best_fit = float("inf")
            best_chrom = None
            best_hist = None

            # Deterministic algorithms only need 1 run
            is_deterministic = alg_name.startswith("Heuristic")
            runs_to_execute = 1 if is_deterministic else num_runs

            for r in range(runs_to_execute):
                # Set local seed for reproducibility and stochastic variance
                np.random.seed(42 + r)
                t_start = time.time()

                # For plotting progress in real-time
                run_generations = []
                run_bests = []
                run_averages = []

                def make_progress_callback(current_alg, run_num, total_runs):
                    def callback(generation, best_fitness, average_fitness, diversity, pc, pm):
                        run_generations.append(generation)
                        run_bests.append(best_fitness)
                        run_averages.append(average_fitness)
                        
                        if generation == 1 or generation % 5 == 0 or generation == max_gens:
                            status_text.markdown(
                                f"🏃 Running **{current_alg}** | Run {run_num}/{total_runs} | Gen {generation}/{max_gens}..."
                            )
                            with realtime_metric_container.container():
                                col_m1, col_m2, col_m3 = st.columns(3)
                                col_m1.metric("Current Run", f"{current_alg} ({run_num}/{total_runs})")
                                col_m2.metric("Generation", f"{generation} / {max_gens}")
                                col_m3.metric("Best Fitness", f"{best_fitness:.4f}")
                            
                            import pandas as pd
                            chart_df = pd.DataFrame({
                                "Generation": run_generations,
                                "Best Fitness": run_bests,
                                "Average Fitness": run_averages
                            }).set_index("Generation")
                            realtime_chart_container.line_chart(chart_df)
                    return callback

                cb = None if is_deterministic else make_progress_callback(alg_name, r + 1, runs_to_execute)

                if alg_name == "Heuristic (FCFS)":
                    sched = HeuristicScheduler(problem_instance, strategy="FCFS")
                    res, chrom, hist = sched.run()
                elif alg_name == "Heuristic (EDD)":
                    sched = HeuristicScheduler(problem_instance, strategy="EDD")
                    res, chrom, hist = sched.run()
                elif alg_name == "Heuristic (SPT)":
                    sched = HeuristicScheduler(problem_instance, strategy="SPT")
                    res, chrom, hist = sched.run()
                elif alg_name == "Heuristic (LPT)":
                    sched = HeuristicScheduler(problem_instance, strategy="LPT")
                    res, chrom, hist = sched.run()
                elif alg_name == "Standard GA":
                    sched = HFGA_TS(problem_instance, use_flc=False, use_tabu=False)
                    res, chrom, hist = sched.run(progress_callback=cb)
                elif alg_name == "GA-Tabu":
                    sched = HFGA_TS(problem_instance, use_flc=False, use_tabu=True)
                    res, chrom, hist = sched.run(progress_callback=cb)
                elif alg_name == "FLC-GA":
                    sched = HFGA_TS(problem_instance, use_flc=True, use_tabu=False)
                    res, chrom, hist = sched.run(progress_callback=cb)
                elif alg_name == "SSO":
                    sched = SSOScheduler(problem_instance)
                    res, chrom, hist = sched.run(progress_callback=cb)
                elif alg_name == "HFGA-TS":
                    sched = HFGA_TS(problem_instance, use_flc=True, use_tabu=True)
                    res, chrom, hist = sched.run(progress_callback=cb)
                else:
                    continue

                duration = time.time() - t_start

                # Track best run based on fitness
                fit = float(alpha * res.total_tardiness + beta * res.makespan)
                
                run_kpis = {
                    "makespan": float(res.makespan),
                    "total_tardiness": float(res.total_tardiness),
                    "tardy_units": int(getattr(res, "total_tardy_units", 0)),
                    "total_setup_cost": float(res.total_setup_cost),
                    "total_setup_time": float(res.total_setup_time),
                    "duration": duration,
                    "fitness": fit,
                }

                runs_data.append(run_kpis)

                if fit < best_fit:
                    best_fit = fit
                    best_res = res
                    best_chrom = chrom
                    best_hist = hist

            # Duplicate deterministic results to keep arrays of size R for statistics
            if is_deterministic and num_runs > 1:
                for _ in range(num_runs - 1):
                    runs_data.append(runs_data[0].copy())

            st.session_state["run_results"]["algs"][alg_name] = {
                "best_result": best_res,
                "best_chrom": best_chrom,
                "best_history": best_hist,
                "runs": runs_data,
            }

            task_idx += 1
            progress_bar.progress(task_idx / total_tasks)

        # Auto-save experiment results
        try:
            saved_folder = save_experiment_data(
                dataset_name, params or {}, st.session_state["run_results"]
            )
            st.session_state["saved_folder"] = saved_folder
        except Exception as e:
            st.error(f"Failed to auto-save experiment: {e}")

        status_text.success(
            "🎉 All comparative optimization experiments completed successfully!"
        )
        st.session_state["optimized"] = True
        st.rerun()
    else:
        if "optimized" not in st.session_state or not st.session_state["optimized"]:
            st.info(
                "Select the algorithms to compare above, set the number of runs, and click the button to start the experiments."
            )
        else:
            st.success(
                "Previous optimization results loaded! Switch to other tabs to view schedule layouts and comparative charts."
            )
            if "saved_folder" in st.session_state:
                st.info(f"📂 Last run saved to: `{st.session_state['saved_folder']}`")
