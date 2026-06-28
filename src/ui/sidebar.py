import streamlit as st
from typing import Dict, Any
from src.config import GA_PARAMETERS


def render_sidebar_controls() -> Dict[str, Any]:
    """Renders SMT/GA parameters in the sidebar grouped by method and returns them as a dictionary."""
    st.sidebar.markdown("### ⚙️ Shared Settings")

    split_threshold = st.sidebar.slider(
        "Batch Splitting Threshold (T)",
        min_value=50,
        max_value=500,
        value=int(GA_PARAMETERS.Threshold_of_Batch_Splitting),
        help="Nominal processing time threshold. Jobs exceeding this are split into sub-lots.",
    )

    st.sidebar.markdown("**Fitness Weights**")
    fitness_alpha = st.sidebar.slider(
        "Tardiness Weight (Alpha)",
        min_value=0.0,
        max_value=1.0,
        value=float(getattr(GA_PARAMETERS, "fitness_alpha", 0.8)),
        step=0.05,
        help="Weight assigned to tardiness in the multi-objective fitness function.",
    )
    fitness_beta = round(1.0 - fitness_alpha, 2)
    st.sidebar.caption(f"Makespan Weight (Beta): **{fitness_beta}**")

    use_parallel = st.sidebar.checkbox(
        "Run in Parallel (Multiprocessing)",
        value=bool(GA_PARAMETERS.Use_Parallel_Execution),
        help="Enable multiple CPU cores to accelerate solution evaluations.",
    )

    # 1. Standard GA parameters
    with st.sidebar.expander("🧬 Genetic Algorithm (GA)", expanded=True):
        pop_size = st.slider(
            "Population Size",
            min_value=10,
            max_value=200,
            value=int(GA_PARAMETERS.Population_Size),
        )
        max_gens = st.slider(
            "Max Generations",
            min_value=10,
            max_value=2000,
            value=int(GA_PARAMETERS.Maximum_Generation),
        )
        elitism_rate = st.slider(
            "Elitism Rate",
            min_value=0.0,
            max_value=0.5,
            value=float(GA_PARAMETERS.Elitism_Rate),
        )
        initial_pc = st.slider(
            "Initial Crossover Rate (Pc)",
            min_value=0.1,
            max_value=1.0,
            value=float(getattr(GA_PARAMETERS, "Initial_Crossover_Rate", 0.8)),
        )
        initial_pm = st.slider(
            "Initial Mutation Rate (Pm)",
            min_value=0.001,
            max_value=0.5,
            value=float(getattr(GA_PARAMETERS, "Initial_Mutation_Rate", 0.1)),
        )

    # 2. Fuzzy Logic bounds
    with st.sidebar.expander("🧠 Fuzzy Logic Controller (FLC)", expanded=False):
        pc_min, pc_max = st.slider(
            "Crossover Rate bounds (Pc)",
            min_value=0.2,
            max_value=1.0,
            value=(
                float(GA_PARAMETERS.Crossover_Rate_Bounds[0]),
                float(GA_PARAMETERS.Crossover_Rate_Bounds[1]),
            ),
        )
        pm_min, pm_max = st.slider(
            "Mutation Rate bounds (Pm)",
            min_value=0.01,
            max_value=0.4,
            value=(
                float(GA_PARAMETERS.Mutation_Rate_Bounds[0]),
                float(GA_PARAMETERS.Mutation_Rate_Bounds[1]),
            ),
        )

    # 3. Tabu Search Neighborhood parameters
    with st.sidebar.expander("🔍 Tabu Search (TS)", expanded=False):
        stall_limit = st.slider(
            "Stall Limit (Triggers TS)",
            min_value=5,
            max_value=500,
            value=int(GA_PARAMETERS.No_Improvement_Limit),
            help="Generations without improvement before triggering local neighborhood search.",
        )
        ts_iter = st.slider(
            "TS Max Iterations",
            min_value=5,
            max_value=150,
            value=int(GA_PARAMETERS.Max_Iterations_of_Tabu_Search),
        )
        tabu_size = st.slider(
            "Tabu List Size",
            min_value=2,
            max_value=100,
            value=int(GA_PARAMETERS.Tabu_List_Size),
        )

    # 4. SSO Swarm parameters
    with st.sidebar.expander("🐝 Simplified Swarm Optimization (SSO)", expanded=False):
        sso_cw = st.slider(
            "Inertia Weight (Cw)",
            min_value=0.0,
            max_value=1.0,
            value=float(getattr(GA_PARAMETERS, "SSO_Cw", 0.20)),
            step=0.05,
            help="Probability to retain the particle's own previous value.",
        )
        sso_cp = st.slider(
            "Pbest Weight (Cp)",
            min_value=0.0,
            max_value=1.0,
            value=float(getattr(GA_PARAMETERS, "SSO_Cp", 0.20)),
            step=0.05,
            help="Probability to copy values from the particle's personal best.",
        )
        sso_cg = st.slider(
            "Gbest Weight (Cg)",
            min_value=0.0,
            max_value=1.0,
            value=float(getattr(GA_PARAMETERS, "SSO_Cg", 0.50)),
            step=0.05,
            help="Probability to copy values from the swarm's global best.",
        )
        sso_cr = st.slider(
            "Random Weight (Cr)",
            min_value=0.0,
            max_value=1.0,
            value=float(getattr(GA_PARAMETERS, "SSO_Cr", 0.10)),
            step=0.05,
            help="Probability to assign a new random value.",
        )

        sso_sum = sso_cw + sso_cp + sso_cg + sso_cr
        if abs(sso_sum - 1.0) > 1e-4:
            st.warning(
                f"⚠️ Probabilities sum to {sso_sum:.2f} (will be auto-normalized to 1.0 during optimization)."
            )

    return {
        "pop_size": pop_size,
        "max_gens": max_gens,
        "elitism_rate": elitism_rate,
        "initial_pc": initial_pc,
        "initial_pm": initial_pm,
        "stall_limit": stall_limit,
        "split_threshold": split_threshold,
        "pc_bounds": [pc_min, pc_max],
        "pm_bounds": [pm_min, pm_max],
        "ts_iter": ts_iter,
        "tabu_size": tabu_size,
        "use_parallel": use_parallel,
        "fitness_alpha": fitness_alpha,
        "fitness_beta": fitness_beta,
        "sso_cw": sso_cw,
        "sso_cp": sso_cp,
        "sso_cg": sso_cg,
        "sso_cr": sso_cr,
    }
