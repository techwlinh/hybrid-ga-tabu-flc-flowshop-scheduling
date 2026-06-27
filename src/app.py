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
import pickle

# Adjust paths to import from src
from src.data.loader import SMTDataLoader
from src.config import GA_PARAMETERS
from src.ui.sync import get_saved_versions, get_sorted_problem_files
from src.ui.sidebar import render_sidebar_controls
from src.ui.tabs import (
    render_dataset_tab,
    render_optimization_tab,
    render_gantt_tab,
    render_explainer_tab,
)

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
    }
</style>
""",
    unsafe_allow_html=True,
)


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

    # Render sidebar parameters and update global config
    params = render_sidebar_controls()
    GA_PARAMETERS.Population_Size = params["pop_size"]
    GA_PARAMETERS.Maximum_Generation = params["max_gens"]
    GA_PARAMETERS.Elitism_Rate = params["elitism_rate"]
    GA_PARAMETERS.No_Improvement_Limit = params["stall_limit"]
    GA_PARAMETERS.Threshold_of_Batch_Splitting = params["split_threshold"]
    GA_PARAMETERS.Crossover_Rate_Bounds = params["pc_bounds"]
    GA_PARAMETERS.Mutation_Rate_Bounds = params["pm_bounds"]
    GA_PARAMETERS.Max_Iterations_of_Tabu_Search = params["ts_iter"]
    GA_PARAMETERS.Tabu_List_Size = params["tabu_size"]
    GA_PARAMETERS.Use_Parallel_Execution = params["use_parallel"]

    # Tabs layout
    tab_data, tab_opt, tab_gantt, tab_explain = st.tabs(
        [
            "📊 Dataset Exploration",
            "⚡ Optimization Run",
            "📈 Interactive Gantt Chart",
            "🧠 Solution Explainer",
        ]
    )

    with tab_data:
        render_dataset_tab(problem_instance)

    with tab_opt:
        render_optimization_tab(problem_instance, params["max_gens"])

    with tab_gantt:
        render_gantt_tab(problem_instance)

    with tab_explain:
        render_explainer_tab(problem_instance)


if __name__ == "__main__":
    main()
