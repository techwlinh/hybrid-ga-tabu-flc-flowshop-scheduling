import numpy as np
import pandas as pd
import streamlit as st
from typing import Any

from src.visualization.gantt import SMTGanttChart
from src.ui.sync import explain_solution


def render_schedule_results_tab(problem_instance: Any):
    """Renders Tab 3: Scheduling Results showing Gantt and tables for each algorithm."""
    st.subheader("📅 Schedule Timeline & Detailed Dispatching Log")

    if "optimized" in st.session_state and st.session_state["optimized"]:
        results = st.session_state["run_results"]
        selected_algs = results["selected_algs"]
        batches = st.session_state["batches"]
        jobs_dict = st.session_state["jobs_dict"]

        # Render sub-tabs for each selected algorithm
        alg_subtabs = st.tabs(selected_algs)

        for idx, alg_name in enumerate(selected_algs):
            with alg_subtabs[idx]:
                alg_data = results["algs"][alg_name]
                res = alg_data["best_result"]

                st.markdown(f"### Best Timeline Configuration for **{alg_name}**")

                # Metrics dashboard
                k_col1, k_col2, k_col3, k_col4, k_col5 = st.columns(5)
                k_col1.metric("Makespan", f"{res.makespan:.1f} min")
                k_col2.metric("Total Tardiness", f"{res.total_tardiness:.1f} min")
                k_col3.metric("Total Setup Cost", f"${res.total_setup_cost:.1f}")
                k_col4.metric("Total Setup Time", f"{res.total_setup_time:.1f} min")
                avg_util = (
                    np.mean(list(res.machine_utilization.values())) * 100
                    if res.machine_utilization
                    else 0.0
                )
                k_col5.metric("Avg Machine Util", f"{avg_util:.1f}%")

                # Time Scale Slider
                max_makespan = float(res.makespan)
                slider_max = max(10.0, float(np.ceil(max_makespan)))

                zoom_limit = st.slider(
                    "Adjust Timeline View Limit (minutes)",
                    min_value=10.0,
                    max_value=slider_max,
                    value=slider_max,
                    step=max(1.0, round(slider_max / 100.0, 1)),
                    help="Adjust the maximum time value shown on the timeline.",
                    key=f"zoom_{alg_name}",
                )

                # Plotly Gantt
                plotly_fig = SMTGanttChart.plot_interactive_gantt(
                    res,
                    jobs_dict,
                    problem_instance.workstations,
                    transport_matrix=getattr(
                        problem_instance, "transport_matrix", None
                    ),
                    max_time_scale=zoom_limit,
                )
                st.plotly_chart(plotly_fig, use_container_width=True)

                # Detailed schedule entries table
                st.write("#### Detailed Schedule Log Entries")
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

                filter_job = st.multiselect(
                    "Filter by Job ID",
                    options=sorted(list(set(df_sched["Job ID"]))),
                    key=f"filter_{alg_name}",
                )
                if filter_job:
                    df_sched = df_sched[df_sched["Job ID"].isin(filter_job)]

                st.dataframe(df_sched, use_container_width=True)

                # Explainer Report for this algorithm
                st.write("---")
                st.write("#### 🧠 Solution Explainer & Diagnostic Report")

                jobs = problem_instance.jobs
                workstations = problem_instance.workstations
                analysis = explain_solution(res, jobs, workstations, batches)

                col_exp1, col_exp2 = st.columns(2)
                with col_exp1:
                    st.markdown(f"""
                    **Manufacturing Bottleneck Analysis:**
                    - **Primary Bottleneck Stage:** **{analysis["bottleneck_name"]}**
                    - **Average Node Utilization:** **{analysis["bottleneck_util"]:.1f}%**
                    
                    > **Bottleneck Explanation:**
                    > {analysis["bottleneck_name"]} represents the highest congestion stage. Adding parallel mounters or optimizing setup matrices here will yield the largest makespan benefits.
                    """)

                    st.markdown(f"""
                    **Batch Splitting Preprocessing Efficiency:**
                    - **Jobs Split:** **{analysis["splits_count"]}** jobs were split.
                    """)
                    for detail in analysis["split_details"][:4]:
                        st.markdown(f"- {detail}")
                with col_exp2:
                    st.markdown("**🕒 Critical Delay & Tardy Jobs:**")
                    if not analysis["tardy_jobs"]:
                        st.success(
                            "🎉 Perfect Schedule! Zero jobs are tardy (all due dates met)."
                        )
                    else:
                        st.warning(
                            f"Total of {len(analysis['tardy_jobs'])} jobs failed to meet their due dates."
                        )
                        for tj in analysis["tardy_jobs"][:4]:
                            st.markdown(f"""
                            - **Job {tj["id"]}** (Priority Group {tj["priority"]}):
                              - **Due Date:** {tj["due_date"]:.1f} min | **Completed at:** {tj["completion"]:.1f} min
                              - **Total Tardiness:** **{tj["tardiness"]:.1f} minutes**
                            """)
    else:
        st.info("Run the optimization first to view the timeline results.")
