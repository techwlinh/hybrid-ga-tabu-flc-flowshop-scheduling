import streamlit as st
from typing import Dict, Any
from src.config import GA_PARAMETERS


def render_sidebar_controls() -> Dict[str, Any]:
    """Renders SMT/GA parameters in the sidebar and returns them as a dictionary."""
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

    return {
        "pop_size": pop_size,
        "max_gens": max_gens,
        "elitism_rate": elitism_rate,
        "stall_limit": stall_limit,
        "split_threshold": split_threshold,
        "pc_bounds": [pc_min, pc_max],
        "pm_bounds": [pm_min, pm_max],
        "ts_iter": ts_iter,
        "tabu_size": tabu_size,
        "use_parallel": use_parallel,
    }
