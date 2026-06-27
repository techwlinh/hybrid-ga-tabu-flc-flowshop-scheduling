import os
import pickle
import time
import numpy as np
import pandas as pd
import streamlit as st
from typing import Dict, List, Any

from src.models import Workstation, Machine, Job, ProblemInstance
from src.algorithm.ga import HFGA_TS
from src.visualization.gantt import SMTGanttChart
from src.visualization.stats import (
    get_scheduling_stats_and_charts,
    plot_machine_workload,
    plot_job_performance_scatter,
    plot_job_status_donut,
    plot_workstation_utilization,
)
from src.ui.sync import sync_workstations, sync_jobs, explain_solution


def render_dataset_tab(problem_instance: Any):
    """Renders Tab 1: Dataset Exploration."""
    st.subheader("Dataset Metadata & Parameters")
    
    jobs = problem_instance.jobs
    workstations = problem_instance.workstations
    setup_times = problem_instance.setup_times
    setup_costs = problem_instance.setup_costs

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Jobs (N)", len(jobs))
    with col2:
        st.metric("Total Workstations (M)", len(workstations))
    with col3:
        total_machines = sum(len(ws.machines) for ws in workstations)
        st.metric("Total SMT Lines (Machines)", total_machines)

    st.write("### Workstations Configuration")
    ws_info = []
    for ws in workstations:
        ws_info.append(
            {"Workstation Name": ws.name, "Machine Count": len(ws.machines)}
        )
    ws_df = pd.DataFrame(ws_info)

    edited_ws_df = st.data_editor(
        ws_df,
        num_rows="dynamic",
        column_config={
            "Workstation Name": st.column_config.TextColumn(required=True),
            "Machine Count": st.column_config.NumberColumn(
                min_value=1, max_value=10, step=1, required=True
            ),
        },
        use_container_width=True,
        key="ws_editor",
    )

    # Trigger synchronization for workstations if changed
    ws_needs_sync = False
    if len(edited_ws_df) != len(workstations):
        ws_needs_sync = True
    else:
        for idx, row in edited_ws_df.iterrows():
            if row["Workstation Name"] != workstations[idx].name or row[
                "Machine Count"
            ] != len(workstations[idx].machines):
                ws_needs_sync = True
                break
                
    if ws_needs_sync:
        sync_workstations(edited_ws_df, problem_instance)
        # Update local references
        jobs = problem_instance.jobs
        workstations = problem_instance.workstations
        setup_times = problem_instance.setup_times
        setup_costs = problem_instance.setup_costs

    # Transportation Times matrix
    st.write("### Transportation Times between Workstations (Matrix)")
    ws_names = [ws.name for ws in workstations]
    transport_df = pd.DataFrame(
        problem_instance.transport_matrix, index=ws_names, columns=ws_names
    )
    edited_transport_df = st.data_editor(
        transport_df, use_container_width=True, key="transport_editor"
    )
    problem_instance.transport_matrix = edited_transport_df.values

    st.write(
        "### Jobs Enriched Metadata (Scaled Quantities & Priorities & Eligibility)"
    )
    # Construct columns for all machines in the system
    machine_columns = []
    mach_tuples = []
    for ws in workstations:
        for mach in ws.machines:
            col_name = f"{ws.name} M{mach.id+1}"
            machine_columns.append(col_name)
            mach_tuples.append((ws.id, mach.id, col_name))

    jobs_data = []
    for job in jobs:
        row = {
            "Job ID": job.id,
            "Quantity (units)": job.quantity,
            "Priority Group": job.priority,
            "Material Arrival (min)": round(job.material_arrival_time, 2),
            "Due Date (min)": round(job.due_date, 2),
        }
        for ws_id, mach_id, col_name in mach_tuples:
            is_eligible = mach_id in job.eligible_machines.get(ws_id, [])
            row[col_name] = is_eligible
        jobs_data.append(row)

    jobs_df = pd.DataFrame(jobs_data)

    # Build column configs
    col_config = {
        "Job ID": st.column_config.NumberColumn(disabled=True),
        "Quantity (units)": st.column_config.NumberColumn(
            min_value=1, required=True
        ),
        "Priority Group": st.column_config.SelectboxColumn(
            options=[1, 2, 3, 4], required=True
        ),
        "Material Arrival (min)": st.column_config.NumberColumn(
            min_value=0.0, required=True
        ),
        "Due Date (min)": st.column_config.NumberColumn(
            min_value=0.0, required=True
        ),
    }
    for col_name in machine_columns:
        col_config[col_name] = st.column_config.CheckboxColumn(required=True)

    edited_jobs_df = st.data_editor(
        jobs_df,
        column_config=col_config,
        use_container_width=True,
        key="jobs_editor",
    )

    # Trigger synchronization for jobs
    sync_jobs(edited_jobs_df, problem_instance, mach_tuples)

    # Setup matrix configuration
    st.write("### Sequence-Dependent Changeover Setup Matrix")
    ws_select = st.selectbox(
        "Select Workstation Stage for Setup Matrix",
        options=workstations,
        format_func=lambda w: w.name,
    )

    if ws_select is not None:
        w_id = ws_select.id
        job_ids = [f"Job {j.id}" for j in jobs]

        col_setup1, col_setup2 = st.columns(2)
        with col_setup1:
            st.write(f"**Setup Times (minutes) at {ws_select.name}**")
            setup_times_df = pd.DataFrame(
                setup_times[:, :, w_id], index=job_ids, columns=job_ids
            )
            edited_setup_times = st.data_editor(
                setup_times_df, use_container_width=True, key=f"setup_times_{w_id}"
            )
            setup_times[:, :, w_id] = edited_setup_times.values

        with col_setup2:
            st.write(f"**Setup Costs ($) at {ws_select.name}**")
            setup_costs_df = pd.DataFrame(
                setup_costs[:, :, w_id], index=job_ids, columns=job_ids
            )
            edited_setup_costs = st.data_editor(
                setup_costs_df, use_container_width=True, key=f"setup_costs_{w_id}"
            )
            setup_costs[:, :, w_id] = edited_setup_costs.values

    # Save version section
    st.write("---")
    st.write("### 💾 Save Current Dataset as a Version")
    with st.form("save_version_form"):
        col_vname, col_vbtn = st.columns([3, 1])
        with col_vname:
            version_name = st.text_input(
                "Enter Version Name",
                value="",
                placeholder="e.g. s1_v1_modified",
                help="Filename under which to save this customized problem instance.",
            ).strip()
        with col_vbtn:
            st.write("")  # Spacer
            st.write("")  # Spacer
            submit_save = st.form_submit_button("Save Dataset Version")

        if submit_save:
            if not version_name:
                st.error("Please enter a valid version name.")
            else:
                safe_name = "".join(
                    c for c in version_name if c.isalnum() or c in ("-", "_")
                ).strip()
                if not safe_name:
                    st.error(
                        "Invalid version name. Use alphanumeric characters, dashes, and underscores."
                    )
                else:
                    filename = f"{safe_name}.pkl"
                    filepath = os.path.join("data_versions", filename)
                    try:
                        with open(filepath, "wb") as f:
                            pickle.dump(problem_instance, f)
                        st.success(
                            f"Successfully saved version as `{filename}` in `data_versions/`!"
                        )
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error saving version: {e}")


def render_optimization_tab(problem_instance: Any, max_gens: int):
    """Renders Tab 2: Optimization Run."""
    st.subheader("Run Hybrid Fuzzy Genetic Algorithm with Tabu Search")

    # Optimization control button
    start_opt = st.button("🚀 Start Optimization Process")

    # Training progress placeholders
    progress_bar = st.progress(0)
    status_text = st.empty()

    # Placeholders for dynamic charts
    col_chart1, col_chart2 = st.columns(2)
    chart_fit = col_chart1.empty()
    chart_flc = col_chart2.empty()

    if start_opt:
        # Initialize GA engine
        ga_engine = HFGA_TS(problem_instance)

        # Setup dynamic data loggers
        history_data = {
            "Generation": [],
            "Best Fitness (Tardiness)": [],
            "Average Fitness": [],
            "Diversity": [],
            "Pc": [],
            "Pm": [],
        }

        def ga_callback(
            generation, best_fitness, average_fitness, diversity, pc, pm
        ):
            # Update progress bar
            prog = min(generation / max_gens, 1.0)
            progress_bar.progress(prog)
            status_text.text(
                f"Generation {generation}/{max_gens} | Best Fitness: {best_fitness:.2f} min | Diversity: {diversity:.3f}"
            )

            # Log data
            history_data["Generation"].append(generation)
            history_data["Best Fitness (Tardiness)"].append(best_fitness)
            history_data["Average Fitness"].append(average_fitness)
            history_data["Diversity"].append(diversity)
            history_data["Pc"].append(pc)
            history_data["Pm"].append(pm)

            # Redraw charts every 5 generations to save rendering lag
            if generation % 5 == 0 or generation == max_gens:
                df = pd.DataFrame(history_data)

                # 1. Fitness curve
                chart_fit.line_chart(
                    df.set_index("Generation")[
                        ["Best Fitness (Tardiness)", "Average Fitness"]
                    ]
                )

                # 2. Diversity & FLC Tracking
                chart_flc.line_chart(
                    df.set_index("Generation")[["Diversity", "Pc", "Pm"]]
                )

        # Run GA with callback hook
        with st.spinner("HFGA-TS optimization is running..."):
            t_start = time.time()
            res, best_chrom, raw_hist = ga_engine.run(progress_callback=ga_callback)
            t_duration = time.time() - t_start

        st.success(
            f"Optimization successfully completed in {t_duration:.2f} seconds!"
        )

        # Save optimization result to streamlit session state to share with other tabs
        st.session_state["opt_result"] = res
        st.session_state["ga_engine"] = ga_engine
        st.session_state["optimized"] = True

        # Display KPIs immediately after run
        st.markdown("### Optimization KPIs Summary")
        kpi_col1, kpi_col2, kpi_col3, kpi_col4, kpi_col5 = st.columns(5)
        kpi_col1.metric("Makespan", f"{res.makespan:.1f} min")
        kpi_col2.metric(
            "Total Tardiness",
            f"{res.total_tardiness:.1f} min",
            delta=None,
            delta_color="inverse",
        )
        kpi_col3.metric("Total Setup Cost", f"${res.total_setup_cost:.1f}")
        kpi_col4.metric("Total Setup Time", f"{res.total_setup_time:.1f} min")
        avg_util = np.mean(list(res.machine_utilization.values())) * 100
        kpi_col5.metric("Avg Machine Util", f"{avg_util:.1f}%")
    else:
        if "optimized" not in st.session_state:
            st.info(
                "Click the button above to run the HFGA-TS algorithm and view results."
            )


def render_gantt_tab(problem_instance: Any):
    """Renders Tab 3: Interactive Gantt Chart."""
    st.subheader("Scheduling Timeline Gantt Visualization")
    if "optimized" in st.session_state and st.session_state["optimized"]:
        res = st.session_state["opt_result"]
        ga_engine = st.session_state["ga_engine"]
        workstations = problem_instance.workstations

        # Time Scale Slider controls
        max_makespan = float(res.makespan)
        slider_max = max(10.0, float(np.ceil(max_makespan)))

        zoom_limit = st.slider(
            "Adjust Timeline View Limit (minutes)",
            min_value=10.0,
            max_value=slider_max,
            value=slider_max,
            step=max(1.0, round(slider_max / 100.0, 1)),
            help="Adjust the maximum time value shown on the timeline.",
        )

        # Display Gantt Chart
        plotly_fig = SMTGanttChart.plot_interactive_gantt(
            res,
            ga_engine.jobs_dict,
            workstations,
            transport_matrix=getattr(problem_instance, "transport_matrix", None),
            max_time_scale=zoom_limit,
        )
        st.plotly_chart(plotly_fig, use_container_width=True)

        # Display detailed tabular search
        st.write("### Detailed Schedule Log Entries")
        df_entries = []
        for entry in res.entries:
            df_entries.append(
                {
                    "Batch ID": entry.batch_id,
                    "Job ID": entry.job_id,
                    "Workstation": entry.workstation_id + 1,
                    "Machine": entry.machine_id + 1,
                    "Start Time (min)": round(entry.start_time, 1),
                    "End Time (min)": round(entry.end_time, 1),
                    "Setup Time (min)": round(entry.setup_time, 1),
                    "Setup Cost ($)": round(entry.setup_cost, 1),
                    "Waiting Time (min)": round(entry.waiting_time, 1),
                }
            )
        df_sched = pd.DataFrame(df_entries)

        # Filter controls in dashboard
        filter_job = st.multiselect(
            "Filter by Job ID", options=sorted(list(set(df_sched["Job ID"])))
        )
        if filter_job:
            df_sched = df_sched[df_sched["Job ID"].isin(filter_job)]

        st.dataframe(df_sched, use_container_width=True)

        # Allow downloads
        csv_data = df_sched.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Schedule as CSV",
            data=csv_data,
            file_name="smt_optimized_schedule.csv",
            mime="text/csv",
        )
    else:
        st.info("Run the optimization first to view the interactive Gantt chart.")


def render_explainer_tab(problem_instance: Any):
    """Renders Tab 4: Solution Explainer."""
    st.subheader("🧠 Explainable Optimization Solution Report")
    if "optimized" in st.session_state and st.session_state["optimized"]:
        res = st.session_state["opt_result"]
        ga_engine = st.session_state["ga_engine"]
        jobs = problem_instance.jobs
        workstations = problem_instance.workstations

        # Run explainer logic
        analysis = explain_solution(res, jobs, workstations, ga_engine.batches)

        col_exp1, col_exp2 = st.columns(2)

        with col_exp1:
            st.markdown(f"""
            ### 🚨 Manufacturing Bottleneck Analysis
            - **Primary Bottleneck Stage:** **{analysis["bottleneck_name"]}**
            - **Average Node Utilization:** **{analysis["bottleneck_util"]:.1f}%**
            
            > **Bottleneck Explanation:**
            > {analysis["bottleneck_name"]} represents the highest congestion point in the flow line. 
            > Delays at this stage propagate directly to downstream workstations. To improve makespan further,
            > you should consider adding an additional SMT nozzle/machine at this workstation to distribute load.
            """)

            st.markdown(f"""
            ### 📦 Batch Splitting Preprocessing Efficiency
            - **Jobs Split:** **{analysis["splits_count"]}** jobs were split.
            
            **Split Details:**
            """)
            for detail in analysis["split_details"]:
                st.markdown(f"- {detail}")
            st.markdown("""
            > **Splitting Explanation:**
            > Bending large quantities into smaller, independent batches allowed the algorithm to route sub-lots
            > across parallel mounters in the workstations simultaneously. This parallelization increases line
            > concurrency and dramatically decreases overall flow makespan.
            """)

        with col_exp2:
            st.markdown("### 🕒 Critical Delay & Tardy Jobs Report")
            if not analysis["tardy_jobs"]:
                st.success(
                    "🎉 Perfect Schedule! Zero jobs are tardy (all due dates met)."
                )
            else:
                st.warning(
                    f"Total of {len(analysis['tardy_jobs'])} jobs failed to meet their due dates."
                )
                for tj in analysis["tardy_jobs"]:
                    st.markdown(f"""
                    **Job {tj["id"]}** (Priority Group {tj["priority"]}):
                    - **Due Date:** {tj["due_date"]:.1f} min | **Completed at:** {tj["completion"]:.1f} min
                    - **Total Tardiness:** **{tj["tardiness"]:.1f} minutes**
                    - **Core Reason for Delay:** {tj["reason"]}
                    ---
                    """)

        # New: Statistical performance & Line balancing section
        st.markdown("---")
        st.subheader("📊 Performance Statistics & Line Balancing Analytics")
        
        stats_data = get_scheduling_stats_and_charts(res, jobs, workstations)
        
        st.write("### 📈 Key Performance Indicators (KPIs)")
        m_col1, m_col2, m_col3 = st.columns(3)
        
        lbe_val = stats_data["lbe_overall"]
        si_val = stats_data["si_overall"]
        compliance = 100.0 - stats_data["tardy_rate"]
        
        m_col1.metric(
            "Line Balance Efficiency (LBE)", 
            f"{lbe_val:.1f}%", 
            help="Measures how evenly workload is distributed across all SMT machines. Ideally > 80%."
        )
        m_col2.metric(
            "Line Smoothness Index (SI)", 
            f"{si_val:.1f}%", 
            help="Measures the dispersion of machine utilization. Lower standard deviation indicates more balanced lines. Ideally < 15%."
        )
        m_col3.metric(
            "On-Time Delivery Rate", 
            f"{compliance:.1f}%", 
            help="Percentage of jobs completed on or before their due dates."
        )
        
        st.write("### 🏭 Machine Workload & Line Balancing Analysis")
        st.markdown("""
        This chart visualizes the workload breakdown of each SMT machine into **Processing**, **Setup Changeover**, and **Idle/Non-allocated** times.
        An uneven distribution within a workstation indicates that a machine is being overloaded while parallel machines are under-utilized.
        """)
        
        ws_names = ["All"] + [ws.name for ws in workstations]
        selected_ws = st.selectbox("Filter Machine Workload by Workstation Stage:", ws_names, index=0)
        
        fig_workload = plot_machine_workload(stats_data["df_machines"], selected_ws)
        st.plotly_chart(fig_workload, use_container_width=True)
        
        col_chart1, col_chart2 = st.columns(2)
        
        with col_chart1:
            st.write("### 🕒 Job Delivery vs. Due Date")
            fig_scatter = plot_job_performance_scatter(stats_data["df_jobs"])
            st.plotly_chart(fig_scatter, use_container_width=True)
            st.markdown("""
            > **Interpretation:** The dashed line represents the due date deadline. Jobs appearing **above** this line are tardy (completed after due date), while jobs on or **below** the line are on-time.
            """)
            
        with col_chart2:
            st.write("### 📊 Delivery Compliance Rate")
            fig_donut = plot_job_status_donut(stats_data["df_jobs"])
            st.plotly_chart(fig_donut, use_container_width=True)

        st.write("### 🏗️ Workstation Stage Utilization Analysis")
        fig_ws_util = plot_workstation_utilization(stats_data["ws_utilization"], workstations)
        st.plotly_chart(fig_ws_util, use_container_width=True)
        st.markdown("""
        > **Interpretation:** This bar chart shows the average machine utilization rate for each workstation stage. The stage with the highest average utilization represents the overall line bottleneck (highlighted in **red**).
        """)
    else:
        st.info(
            "Run the optimization first to generate the explainable solution analysis report."
        )
