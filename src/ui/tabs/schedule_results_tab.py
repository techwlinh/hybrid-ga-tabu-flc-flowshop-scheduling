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
                st.plotly_chart(plotly_fig, use_container_width=True, key=f"gantt_chart_{alg_name}")

                # Detailed schedule entries table
                st.write("#### Detailed Schedule Log Entries")
                df_entries = []
                for entry in res.entries:
                    df_entries.append(
                        {
                            "Batch ID": entry.batch_id,
                            "Product ID": entry.job_id,
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
                    "Filter by Product ID",
                    options=sorted(list(set(df_sched["Product ID"]))),
                    key=f"filter_{alg_name}",
                )
                if filter_job:
                    df_sched = df_sched[df_sched["Product ID"].isin(filter_job)]

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
                    - **Products Split:** **{analysis["splits_count"]}** products were split.
                    """)
                    for detail in analysis["split_details"][:4]:
                        st.markdown(f"- {detail}")
                with col_exp2:
                    st.markdown("**🕒 Critical Delay & Tardy Products Summary:**")
                    if not analysis["tardy_jobs"]:
                        st.success(
                            "🎉 Perfect Schedule! Zero products are tardy (all due dates met)."
                        )
                    else:
                        st.warning(
                            f"Total of {len(analysis['tardy_jobs'])} products failed to meet their due dates."
                        )
                        # Create and show the tardy units table for each product ID
                        tardy_records = []
                        for tj in analysis["tardy_jobs"]:
                            job = jobs_dict[tj["id"]]
                            tardy_records.append({
                                "Product ID": tj["id"],
                                "Priority": f"Priority {tj['priority']}",
                                "Total Units": job.quantity,
                                "Tardy Units (pcs)": tj.get("tardy_units", 0),
                                "Due Date (min)": round(tj["due_date"], 1),
                                "Completion (min)": round(tj["completion"], 1),
                                "Tardiness (min)": round(tj["tardiness"], 1),
                                "Reason": tj["reason"]
                            })
                        df_tardy_prods = pd.DataFrame(tardy_records)
                        st.dataframe(df_tardy_prods.set_index("Product ID"), use_container_width=True)

                # 4. Priority Group Tardiness Evaluation & Distribution
                st.write("---")
                st.write("##### 🏷️ Tardiness Breakdown & Distribution by Product Priority Group")

                # Calculate job completion times at final stage
                job_completions = {}
                num_ws = len(workstations)
                for entry in res.entries:
                    if entry.workstation_id == num_ws - 1:
                        job_completions[entry.job_id] = max(
                            job_completions.get(entry.job_id, 0.0), entry.end_time
                        )

                priorities_list = sorted(list(set(job.priority for job in jobs)))
                priority_stats = {}
                for p in priorities_list:
                    priority_stats[p] = {"count_tardy": 0, "total_tardiness": 0.0, "total_jobs": 0, "tardy_units": 0, "total_units": 0}

                # Find batch completion times at final stage
                batch_completions = {}
                for entry in res.entries:
                    if entry.workstation_id == num_ws - 1:
                        batch_completions[entry.batch_id] = entry.end_time

                for job in jobs:
                    comp_time = job_completions.get(job.id, 0.0)
                    tardiness = max(0.0, comp_time - job.due_date)
                    p = job.priority
                    if p in priority_stats:
                        priority_stats[p]["total_jobs"] += 1
                        priority_stats[p]["total_units"] += job.quantity
                        
                        # Calculate tardy units for this job by looking at its late batches
                        tardy_units_job = 0
                        job_batches = [b for b in batches if b.job_id == job.id]
                        for b in job_batches:
                            c_time = batch_completions.get(b.id, 0.0)
                            if c_time > job.due_date:
                                tardy_units_job += b.quantity
                        
                        priority_stats[p]["tardy_units"] += tardy_units_job
                        if tardiness > 0.0:
                            priority_stats[p]["count_tardy"] += 1
                            priority_stats[p]["total_tardiness"] += tardiness

                col_p1, col_p2 = st.columns([2, 3])
                with col_p1:
                    st.write("**Priority Group Summary**")
                    priority_records = []
                    for p in priorities_list:
                        priority_records.append({
                            "Priority Group": f"Priority {p}",
                            "Total Products": priority_stats[p]["total_jobs"],
                            "Total Units": priority_stats[p]["total_units"],
                            "Tardy Units (pcs)": priority_stats[p]["tardy_units"],
                            "Total Tardiness (min)": round(priority_stats[p]["total_tardiness"], 1),
                        })
                    
                    # Append Total row
                    total_products = sum(priority_stats[p]["total_jobs"] for p in priorities_list)
                    total_units = sum(priority_stats[p]["total_units"] for p in priorities_list)
                    total_tardy_units = sum(priority_stats[p]["tardy_units"] for p in priorities_list)
                    sum_tardiness = sum(priority_stats[p]["total_tardiness"] for p in priorities_list)
                    priority_records.append({
                        "Priority Group": "Total",
                        "Total Products": total_products,
                        "Total Units": total_units,
                        "Tardy Units (pcs)": total_tardy_units,
                        "Total Tardiness (min)": round(sum_tardiness, 1),
                    })
                    
                    df_priority = pd.DataFrame(priority_records)
                    st.dataframe(df_priority.set_index("Priority Group"), use_container_width=True)
                
                with col_p2:
                    import plotly.graph_objects as go
                    from plotly.subplots import make_subplots
                    
                    priorities = [f"Priority {p}" for p in priorities_list]
                    tardy_units_vals = [priority_stats[p]["tardy_units"] for p in priorities_list]
                    total_tardiness_vals = [priority_stats[p]["total_tardiness"] for p in priorities_list]
                    
                    fig_p = make_subplots(
                        rows=1, cols=2, 
                        subplot_titles=("Number of Tardy Units (pcs)", "Total Tardiness Time (min)")
                    )
                    
                    fig_p.add_trace(
                        go.Bar(x=priorities, y=tardy_units_vals, marker_color="#EF4444", name="Tardy Units", showlegend=False),
                        row=1, col=1
                    )
                    
                    fig_p.add_trace(
                        go.Bar(x=priorities, y=total_tardiness_vals, marker_color="#F59E0B", name="Tardiness Time", showlegend=False),
                        row=1, col=2
                    )
                    
                    fig_p.update_layout(
                        title_text=f"Priority Tardiness Distribution ({alg_name})",
                        height=280,
                        margin=dict(l=10, r=10, t=40, b=10)
                    )
                    st.plotly_chart(fig_p, use_container_width=True, key=f"priority_dist_{alg_name}")


    else:
        st.info("Run the optimization first to view the timeline results.")
