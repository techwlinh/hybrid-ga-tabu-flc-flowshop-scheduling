import sys
import os
import importlib

# Add the project root directory to sys.path to allow importing from src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Reload local modules to prevent Streamlit from caching old versions
# We only reload the visualization module to avoid breaking class identity for multiprocessing pickles
for module_name in ["src.visualization.gantt"]:
    if module_name in sys.modules:
        importlib.reload(sys.modules[module_name])

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
from src.visualization.stats import (
    get_scheduling_stats_and_charts,
    plot_machine_workload,
    plot_job_performance_scatter,
    plot_job_status_donut,
    plot_workstation_utilization,
)
import pickle
from src.models import Workstation, Machine, Job, ProblemInstance


def get_saved_versions() -> List[str]:
    """Scans data_versions/ and returns sorted .pkl filenames."""
    folder = "data_versions"
    os.makedirs(folder, exist_ok=True)
    files = [f for f in os.listdir(folder) if f.endswith(".pkl")]
    return sorted(files)


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


# Page Configuration for Premium Design
st.set_page_config(
    page_title="HFGA-TS SMT Scheduling System",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for Zinc Minimal Palette & Typography (JetBrains Mono for Data)
st.markdown(
    """
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
""",
    unsafe_allow_html=True,
)


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


def main():
    st.title("🏭 SMT Line Scheduling Optimization Dashboard (HFGA-TS)")
    st.write(
        "An advanced Hybrid Fuzzy Genetic Algorithm with Tabu Search for unrelated parallel SMT machines."
    )

    # Scan problem files
    problem_files = get_sorted_problem_files()
    if not problem_files:
        st.error(
            "No MATLAB problem files found in `refs/Antonella Branda/Problems`. Please check data paths."
        )
        return

    # Sidebar: Configurations
    st.sidebar.header("🔧 Configuration panel")

    # Dataset selection
    st.sidebar.write("### Dataset Selection")
    col_orig, col_saved = st.sidebar.columns(2)

    with col_orig:
        selected_file_path = st.selectbox(
            "Original Dataset (.mat)",
            options=problem_files,
            format_func=lambda x: os.path.basename(x),
        )

    with col_saved:
        saved_versions = get_saved_versions()
        selected_version = st.selectbox(
            "Saved Data Version (.pkl)", options=["None"] + saved_versions
        )

    # State key for reloading detection
    current_key = f"{selected_file_path}__{selected_version}"

    # Initialize active_problem state
    if (
        "last_loaded_key" not in st.session_state
        or st.session_state["last_loaded_key"] != current_key
        or "active_problem" not in st.session_state
    ):
        st.session_state["last_loaded_key"] = current_key
        st.session_state["optimized"] = False
        if "opt_result" in st.session_state:
            del st.session_state["opt_result"]
        if "ga_engine" in st.session_state:
            del st.session_state["ga_engine"]

        with st.spinner("Loading dataset..."):
            if selected_version != "None":
                filepath = os.path.join("data_versions", selected_version)
                try:
                    with open(filepath, "rb") as f:
                        problem_instance = pickle.load(f)
                except Exception as e:
                    st.error(
                        f"Failed to load saved version: {e}. Falling back to original."
                    )
                    problem_instance = SMTDataLoader.load_mat_problem(
                        selected_file_path
                    )
            else:
                problem_instance = SMTDataLoader.load_mat_problem(selected_file_path)

            st.session_state["active_problem"] = problem_instance

    # Always read active problem from session state
    problem_instance = st.session_state["active_problem"]

    # Safety fallback: initialize transport_matrix dynamically if it is missing (due to caching/pickle)
    if (
        not hasattr(problem_instance, "transport_matrix")
        or problem_instance.transport_matrix is None
    ):
        num_ws = len(problem_instance.workstations)
        import hashlib

        file_hash = (
            int(hashlib.md5(selected_file_path.encode("utf-8")).hexdigest(), 16) % 10000
        )
        prng = np.random.RandomState(42 + file_hash)
        transport_matrix = prng.uniform(5.0, 15.0, size=(num_ws, num_ws))
        transport_matrix = np.round(transport_matrix, 1)
        np.fill_diagonal(transport_matrix, 0.0)
        problem_instance.transport_matrix = transport_matrix

    jobs = problem_instance.jobs
    workstations = problem_instance.workstations
    setup_times = problem_instance.setup_times
    setup_costs = problem_instance.setup_costs

    # 2. Hyperparameter controls in Sidebar
    st.sidebar.subheader("GA Parameters")
    pop_size = st.sidebar.slider(
        "Population Size",
        min_value=10,
        max_value=200,
        value=int(GA_PARAMETERS["Population_Size"]),
    )
    max_gens = st.sidebar.slider(
        "Max Generations",
        min_value=50,
        max_value=2000,
        value=int(GA_PARAMETERS["Maximum_Generation"]),
    )
    elitism_rate = st.sidebar.slider(
        "Elitism Rate",
        min_value=0.01,
        max_value=0.3,
        value=float(GA_PARAMETERS["Elitism_Rate"]),
    )
    stall_limit = st.sidebar.slider(
        "Stall Limit (for TS)",
        min_value=10,
        max_value=200,
        value=int(GA_PARAMETERS["No_Improvement_Limit"]),
    )
    split_threshold = st.sidebar.slider(
        "Batch Splitting Threshold (T)",
        min_value=50,
        max_value=500,
        value=int(GA_PARAMETERS["Threshold_of_Batch_Splitting"]),
    )

    st.sidebar.subheader("FLC Bounds")
    pc_min, pc_max = st.sidebar.slider(
        "Crossover Rate bounds (Pc)",
        min_value=0.2,
        max_value=1.0,
        value=(
            float(GA_PARAMETERS["Crossover_Rate_Bounds"][0]),
            float(GA_PARAMETERS["Crossover_Rate_Bounds"][1]),
        ),
    )
    pm_min, pm_max = st.sidebar.slider(
        "Mutation Rate bounds (Pm)",
        min_value=0.01,
        max_value=0.4,
        value=(
            float(GA_PARAMETERS["Mutation_Rate_Bounds"][0]),
            float(GA_PARAMETERS["Mutation_Rate_Bounds"][1]),
        ),
    )

    st.sidebar.subheader("Tabu Search Settings")
    ts_iter = st.sidebar.slider(
        "TS Max Iterations",
        min_value=5,
        max_value=100,
        value=int(GA_PARAMETERS["Max_Iterations_of_Tabu_Search"]),
    )
    tabu_size = st.sidebar.slider(
        "Tabu List Size",
        min_value=5,
        max_value=50,
        value=int(GA_PARAMETERS["Tabu_List_Size"]),
    )

    st.sidebar.subheader("Performance Optimization")
    use_parallel = st.sidebar.checkbox(
        "Run in Parallel (Multiprocessing)",
        value=bool(GA_PARAMETERS["Use_Parallel_Execution"]),
    )

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
    GA_PARAMETERS["Use_Parallel_Execution"] = use_parallel

    # Tabs layout
    tab_data, tab_opt, tab_gantt, tab_explain = st.tabs(
        [
            "📊 Dataset Exploration",
            "⚡ Optimization Run",
            "📈 Interactive Gantt Chart",
            "🧠 Solution Explainer",
        ]
    )

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
            # Re-read configurations
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

    # ------------------ Tab 3: Interactive Gantt Chart ------------------
    with tab_gantt:
        st.subheader("Scheduling Timeline Gantt Visualization")
        if "optimized" in st.session_state and st.session_state["optimized"]:
            res = st.session_state["opt_result"]
            ga_engine = st.session_state["ga_engine"]

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


if __name__ == "__main__":
    main()
