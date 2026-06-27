import sys
import os
# Add the project root directory to sys.path to allow importing from src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import numpy as np
import pandas as pd
import time
from typing import Dict, List, Any

# Adjust paths to import from src
from src.data.loader import SMTDataLoader
from src.algorithm.ga import HFGA_TS
from src.config import GA_PARAMETERS, SMT_PARAMETERS
from src.visualization.gantt import SMTGanttChart

# Page Configuration for Premium Design
st.set_page_config(
    page_title="HFGA-TS SMT Scheduling System",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Zinc Minimal Palette & Typography (JetBrains Mono for Data)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,100..1000;1,9..40,100..1000&family=JetBrains+Mono:ital,wght@0,100..800;1,100..800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }
    
    .stMetric {
        background-color: #FAFAFA;
        border: 1px solid #E4E4E7;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.02);
    }
    
    code, pre, .mono-text {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 13px !important;
    }
    
    .stButton>button {
        background-color: #18181B !important;
        color: #FFFFFF !important;
        border-radius: 6px !important;
        border: none !important;
        padding: 8px 16px !important;
        font-weight: 500 !important;
    }
    .stButton>button:hover {
        background-color: #27272A !important;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        padding-left: 0px;
        padding-right: 0px;
        font-weight: 600;
        font-size: 14px;
        color: #71717A;
    }
    .stTabs [aria-selected="true"] {
        color: #18181B !important;
</style>
""", unsafe_allow_html=True)

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

def explain_solution(result: Any, jobs: List[Any], workstations: List[Any], batches: List[Any]) -> Dict[str, Any]:
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
            job_completions[entry.job_id] = max(job_completions.get(entry.job_id, 0.0), entry.end_time)
            
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
                job_setup_entries = [e for e in result.entries if e.job_id == job.id and e.setup_time > 0]
                total_job_setup = sum(e.setup_time for e in job_setup_entries)
                if total_job_setup > 20:
                    reason = f"High changeover/setup time overhead ({total_job_setup:.1f} min spent in setups across lines)"
                else:
                    reason = "High volume processing bottleneck at workstations"
                    
            tardy_jobs.append({
                "id": job.id,
                "due_date": job.due_date,
                "completion": comp_time,
                "tardiness": tardiness,
                "priority": job.priority,
                "reason": reason
            })
            
    # 3. Batch splitting summary
    splits_count = 0
    job_batches = {}
    for b in batches:
        job_batches[b.job_id] = job_batches.get(b.job_id, 0) + 1
    
    split_details = []
    for j_id, count in job_batches.items():
        if count > 1:
            splits_count += 1
            split_details.append(f"Job {j_id} split into {count} batches to distribute its quantity of {jobs[j_id].quantity} units")
            
    return {
        "bottleneck_id": bottleneck_ws_id,
        "bottleneck_name": bottleneck_ws_name,
        "bottleneck_util": ws_util[bottleneck_ws_id] * 100,
        "tardy_jobs": tardy_jobs,
        "splits_count": splits_count,
        "split_details": split_details[:5] # Top 5 details
    }

def main():
    st.title("🏭 SMT Line Scheduling Optimization Dashboard (HFGA-TS)")
    st.write("An advanced Hybrid Fuzzy Genetic Algorithm with Tabu Search for unrelated parallel SMT machines.")
    
    # Scan problem files
    problem_files = get_sorted_problem_files()
    if not problem_files:
        st.error("No MATLAB problem files found in `refs/Antonella Branda/Problems`. Please check data paths.")
        return
        
    # Sidebar: Configurations
    st.sidebar.header("🔧 Configuration panel")
    
    # Dataset selection
    selected_file_path = st.sidebar.selectbox(
        "Select Problem Dataset (.mat)",
        options=problem_files,
        format_func=lambda x: os.path.basename(x)
    )
    
    # 1. Load data
    with st.spinner("Loading dataset..."):
        problem_instance = SMTDataLoader.load_mat_problem(selected_file_path)
        jobs = problem_instance.jobs
        workstations = problem_instance.workstations
        setup_times = problem_instance.setup_times
        setup_costs = problem_instance.setup_costs
        
    # 2. Hyperparameter controls in Sidebar
    st.sidebar.subheader("GA Parameters")
    pop_size = st.sidebar.slider("Population Size", min_value=10, max_value=200, value=int(GA_PARAMETERS["Population_Size"]))
    max_gens = st.sidebar.slider("Max Generations", min_value=50, max_value=2000, value=int(GA_PARAMETERS["Maximum_Generation"]))
    elitism_rate = st.sidebar.slider("Elitism Rate", min_value=0.01, max_value=0.3, value=float(GA_PARAMETERS["Elitism_Rate"]))
    stall_limit = st.sidebar.slider("Stall Limit (for TS)", min_value=10, max_value=200, value=int(GA_PARAMETERS["No_Improvement_Limit"]))
    split_threshold = st.sidebar.slider("Batch Splitting Threshold (T)", min_value=50, max_value=500, value=int(GA_PARAMETERS["Threshold_of_Batch_Splitting"]))
    
    st.sidebar.subheader("FLC Bounds")
    pc_min, pc_max = st.sidebar.slider("Crossover Rate bounds (Pc)", min_value=0.2, max_value=1.0, value=(float(GA_PARAMETERS["Crossover_Rate_Bounds"][0]), float(GA_PARAMETERS["Crossover_Rate_Bounds"][1])))
    pm_min, pm_max = st.sidebar.slider("Mutation Rate bounds (Pm)", min_value=0.01, max_value=0.4, value=(float(GA_PARAMETERS["Mutation_Rate_Bounds"][0]), float(GA_PARAMETERS["Mutation_Rate_Bounds"][1])))
    
    st.sidebar.subheader("Tabu Search Settings")
    ts_iter = st.sidebar.slider("TS Max Iterations", min_value=5, max_value=100, value=int(GA_PARAMETERS["Max_Iterations_of_Tabu_Search"]))
    tabu_size = st.sidebar.slider("Tabu List Size", min_value=5, max_value=50, value=int(GA_PARAMETERS["Tabu_List_Size"]))
    
    st.sidebar.subheader("SMT Parameters")
    transport_time = st.sidebar.slider("Workstation Transport Time (min)", min_value=0.0, max_value=30.0, value=float(SMT_PARAMETERS["Default_Transport_Time"]))
    
    st.sidebar.subheader("Performance Optimization")
    use_parallel = st.sidebar.checkbox("Run in Parallel (Multiprocessing)", value=bool(GA_PARAMETERS["Use_Parallel_Execution"]))
    
    # Update global config dynamically based on sidebar
    GA_PARAMETERS["Population_Size"] = pop_size
    GA_PARAMETERS["Maximum_Generation"] = max_gens
    GA_PARAMETERS["Elitism_Rate"] = elitism_rate
    GA_PARAMETERS["No_Improvement_Limit"] = stall_limit
    GA_PARAMETERS["Threshold_of_Batch_Splitting"] = split_threshold
    GA_PARAMETERS["Crossover_Rate_Bounds"] = [pc_min, pc_max]
    GA_PARAMETERS["Mutation_Rate_Bounds"] = [pm_min, pm_max]
    GA_PARAMETERS["Max_Iterations_of_Tabu_Search"] = ts_iter
    GA_PARAMETERS["Tabu_List_Size"] = tabu_size
    SMT_PARAMETERS["Default_Transport_Time"] = transport_time
    GA_PARAMETERS["Use_Parallel_Execution"] = use_parallel

    # Tabs layout
    tab_data, tab_opt, tab_gantt, tab_explain = st.tabs([
        "📊 Dataset Exploration", 
        "⚡ Optimization Run", 
        "📈 Interactive Gantt Chart", 
        "🧠 Solution Explainer"
    ])
    
    # ------------------ Tab 1: Dataset Exploration ------------------
    with tab_data:
        st.subheader("Dataset Metadata & Parameters")
        
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
            ws_info.append({
                "Workstation ID": ws.id + 1,
                "Workstation Name": ws.name,
                "Machine Count": len(ws.machines),
                "Machine Names": ", ".join(m.name for m in ws.machines)
            })
        st.dataframe(pd.DataFrame(ws_info), use_container_width=True)
            
        st.write("### Jobs Enriched Metadata (Scaled Quantities & Priorities)")
        jobs_info = []
        for job in jobs:
            jobs_info.append({
                "Job ID": job.id,
                "Quantity (units)": job.quantity,
                "Priority Group": job.priority,
                "Material Arrival (min)": round(job.material_arrival_time, 2),
                "Due Date (min)": round(job.due_date, 2),
                "Eligible Machines per WS": ", ".join(f"WS{w+1}:{len(m_list)}" for w, m_list in job.eligible_machines.items())
            })
        st.dataframe(pd.DataFrame(jobs_info), use_container_width=True)

    # ------------------ Tab 2: Optimization Run ------------------
    with tab_opt:
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
                "Pm": []
            }
            
            def ga_callback(generation, best_fitness, average_fitness, diversity, pc, pm):
                # Update progress bar
                prog = min(generation / max_gens, 1.0)
                progress_bar.progress(prog)
                status_text.text(f"Generation {generation}/{max_gens} | Best Fitness: {best_fitness:.2f} min | Diversity: {diversity:.3f}")
                
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
                    chart_fit.line_chart(df.set_index("Generation")[["Best Fitness (Tardiness)", "Average Fitness"]])
                    
                    # 2. Diversity & FLC Tracking
                    chart_flc.line_chart(df.set_index("Generation")[["Diversity", "Pc", "Pm"]])
                    
            # Run GA with callback hook
            with st.spinner("HFGA-TS optimization is running..."):
                t_start = time.time()
                res, best_chrom, raw_hist = ga_engine.run(progress_callback=ga_callback)
                t_duration = time.time() - t_start
                
            st.success(f"Optimization successfully completed in {t_duration:.2f} seconds!")
            
            # Save optimization result to streamlit session state to share with other tabs
            st.session_state["opt_result"] = res
            st.session_state["ga_engine"] = ga_engine
            st.session_state["optimized"] = True
            
            # Display KPIs immediately after run
            st.markdown("### Optimization KPIs Summary")
            kpi_col1, kpi_col2, kpi_col3, kpi_col4, kpi_col5 = st.columns(5)
            kpi_col1.metric("Makespan", f"{res.makespan:.1f} min")
            kpi_col2.metric("Total Tardiness", f"{res.total_tardiness:.1f} min", delta=None, delta_color="inverse")
            kpi_col3.metric("Total Setup Cost", f"${res.total_setup_cost:.1f}")
            kpi_col4.metric("Total Setup Time", f"{res.total_setup_time:.1f} min")
            avg_util = np.mean(list(res.machine_utilization.values())) * 100
            kpi_col5.metric("Avg Machine Util", f"{avg_util:.1f}%")
        else:
            if "optimized" not in st.session_state:
                st.info("Click the button above to run the HFGA-TS algorithm and view results.")

    # ------------------ Tab 3: Interactive Gantt Chart ------------------
    with tab_gantt:
        st.subheader("Scheduling Timeline Gantt Visualization")
        if "optimized" in st.session_state and st.session_state["optimized"]:
            res = st.session_state["opt_result"]
            ga_engine = st.session_state["ga_engine"]
            
            # Display Gantt Chart
            plotly_fig = SMTGanttChart.plot_interactive_gantt(res, ga_engine.jobs_dict, workstations)
            st.plotly_chart(plotly_fig, use_container_width=True)
            
            # Display detailed tabular search
            st.write("### Detailed Schedule Log Entries")
            df_entries = []
            for entry in res.entries:
                df_entries.append({
                    "Batch ID": entry.batch_id,
                    "Job ID": entry.job_id,
                    "Workstation": entry.workstation_id + 1,
                    "Machine": entry.machine_id + 1,
                    "Start Time (min)": round(entry.start_time, 1),
                    "End Time (min)": round(entry.end_time, 1),
                    "Setup Time (min)": round(entry.setup_time, 1),
                    "Setup Cost ($)": round(entry.setup_cost, 1),
                    "Waiting Time (min)": round(entry.waiting_time, 1)
                })
            df_sched = pd.DataFrame(df_entries)
            
            # Filter controls in dashboard
            filter_job = st.multiselect("Filter by Job ID", options=sorted(list(set(df_sched["Job ID"]))))
            if filter_job:
                df_sched = df_sched[df_sched["Job ID"].isin(filter_job)]
                
            st.dataframe(df_sched, use_container_width=True)
            
            # Allow downloads
            csv_data = df_sched.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Schedule as CSV",
                data=csv_data,
                file_name="smt_optimized_schedule.csv",
                mime="text/csv"
            )
        else:
            st.info("Run the optimization first to view the interactive Gantt chart.")

    # ------------------ Tab 4: Solution Explainer ------------------
    with tab_explain:
        st.subheader("🧠 Explainable Optimization Solution Report")
        if "optimized" in st.session_state and st.session_state["optimized"]:
            res = st.session_state["opt_result"]
            ga_engine = st.session_state["ga_engine"]
            
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
                    st.success("🎉 Perfect Schedule! Zero jobs are tardy (all due dates met).")
                else:
                    st.warning(f"Total of {len(analysis['tardy_jobs'])} jobs failed to meet their due dates.")
                    for tj in analysis["tardy_jobs"]:
                        st.markdown(f"""
                        **Job {tj["id"]}** (Priority Group {tj["priority"]}):
                        - **Due Date:** {tj["due_date"]:.1f} min | **Completed at:** {tj["completion"]:.1f} min
                        - **Total Tardiness:** **{tj["tardiness"]:.1f} minutes**
                        - **Core Reason for Delay:** {tj["reason"]}
                        ---
                        """)
        else:
            st.info("Run the optimization first to generate the explainable solution analysis report.")

if __name__ == "__main__":
    main()
