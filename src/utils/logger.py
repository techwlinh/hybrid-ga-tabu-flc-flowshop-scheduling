import os
import json
import datetime
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def get_next_experiment_number(experiments_dir="experiments"):
    """
    Finds the next experiment sequence number by scanning the experiments folder.
    """
    os.makedirs(experiments_dir, exist_ok=True)
    existing_dirs = os.listdir(experiments_dir)
    max_num = 0
    for d in existing_dirs:
        # Match folders starting with digits, e.g., 001_..., 1_...
        match = re.match(r"^(\d+)_", d)
        if match:
            try:
                num = int(match.group(1))
                if num > max_num:
                    max_num = num
            except ValueError:
                pass
    return max_num + 1

def save_experiment_data(dataset_name: str, params: dict, run_results: dict, experiments_dir="experiments") -> str:
    """
    Saves the fitness evolution to a CSV file, metadata to a JSON file, and
    plots the evolution using matplotlib, saving it to a folder named:
    experiments/{number}_{time}_{dataname}
    
    Returns the path of the created folder.
    """
    # 1. Prepare directory metadata
    next_num = get_next_experiment_number(experiments_dir)
    number_str = f"{next_num:03d}"
    
    now = datetime.datetime.now()
    time_str = now.strftime("%Y%m%d_%H%M%S")
    
    subfolder_name = f"{number_str}_{time_str}_{dataset_name}"
    subfolder_path = os.path.join(experiments_dir, subfolder_name)
    os.makedirs(subfolder_path, exist_ok=True)
    
    # Define file paths
    csv_path = os.path.join(subfolder_path, f"{number_str}_{time_str}_{dataset_name}.csv")
    json_path = os.path.join(subfolder_path, f"{number_str}_{time_str}_{dataset_name}.json")
    plot_path = os.path.join(subfolder_path, f"{number_str}_{time_str}_{dataset_name}.png")
    
    selected_algs = run_results.get("selected_algs", [])
    
    # 2. Build and save CSV for evolution history
    max_gen = 1
    for alg_name in selected_algs:
        alg_data = run_results.get("algs", {}).get(alg_name, {})
        hist = alg_data.get("best_history", {})
        if hist and "generation" in hist:
            max_gen = max(max_gen, max(hist["generation"], default=1))
            
    csv_data = {"Generation": list(range(1, max_gen + 1))}
    for alg_name in selected_algs:
        alg_data = run_results.get("algs", {}).get(alg_name, {})
        hist = alg_data.get("best_history", {})
        
        fit_vals = []
        if hist and "best_fitness" in hist and len(hist["best_fitness"]) > 0:
            generations = hist.get("generation", [])
            fitnesses = hist.get("best_fitness", [])
            gen_to_fit = dict(zip(generations, fitnesses))
            
            last_val = fitnesses[0]
            for g in range(1, max_gen + 1):
                if g in gen_to_fit:
                    last_val = gen_to_fit[g]
                fit_vals.append(last_val)
        else:
            best_res = alg_data.get("best_result")
            if best_res:
                from src.config import GA_PARAMETERS
                alpha = getattr(GA_PARAMETERS, "fitness_alpha", 0.5)
                beta = getattr(GA_PARAMETERS, "fitness_beta", 0.5)
                val = float(alpha * best_res.total_tardiness + beta * best_res.makespan)
            else:
                val = 0.0
            fit_vals = [val] * max_gen
            
        csv_data[alg_name] = fit_vals
        
    df_evolution = pd.DataFrame(csv_data)
    df_evolution.to_csv(csv_path, index=False)
    
    # 3. Create Matplotlib plot
    plt.figure(figsize=(10, 6))
    
    colors = ['#18181B', '#3B82F6', '#10B981', '#F59E0B', '#EF4444', '#8B5CF6', '#EC4899', '#6B7280']
    
    for i, alg_name in enumerate(selected_algs):
        y_vals = csv_data[alg_name]
        x_vals = csv_data["Generation"]
        color = colors[i % len(colors)]
        
        if alg_name.startswith("Heuristic"):
            plt.plot(x_vals, y_vals, label=alg_name, color=color, linestyle="--", linewidth=1.5)
        else:
            plt.plot(x_vals, y_vals, label=alg_name, color=color, linewidth=2.0)
            
    plt.title(f"Fitness Evolution on {dataset_name}", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Generation / Iteration", fontsize=12, labelpad=10)
    from src.config import GA_PARAMETERS
    alpha = getattr(GA_PARAMETERS, "fitness_alpha", 0.5)
    beta = getattr(GA_PARAMETERS, "fitness_beta", 0.5)
    plt.ylabel(f"Fitness Value ({alpha} * Tardiness + {beta} * Makespan)", fontsize=12, labelpad=10)
    plt.legend(frameon=True, facecolor="white", edgecolor="#E4E4E7", loc="best")
    plt.grid(True, linestyle="--", alpha=0.5, color="#E4E4E7")
    
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#E4E4E7')
    ax.spines['bottom'].set_color('#E4E4E7')
    
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()
    
    # 4. Save metadata JSON
    metadata = {
        "dataset_name": dataset_name,
        "timestamp": time_str,
        "number_of_runs": run_results.get("num_runs", 1),
        "parameters": params,
        "results": {}
    }
    
    for alg_name, alg_data in run_results.get("algs", {}).items():
        runs = alg_data.get("runs", [])
        if runs:
            makespans = [r["makespan"] for r in runs]
            tardinesses = [r["total_tardiness"] for r in runs]
            setup_costs = [r["total_setup_cost"] for r in runs]
            setup_times = [r["total_setup_time"] for r in runs]
            durations = [r["duration"] for r in runs]
            
            metadata["results"][alg_name] = {
                "best_run": {
                    "makespan": float(alg_data["best_result"].makespan),
                    "total_tardiness": float(alg_data["best_result"].total_tardiness),
                    "total_setup_cost": float(alg_data["best_result"].total_setup_cost),
                    "total_setup_time": float(alg_data["best_result"].total_setup_time),
                },
                "statistics": {
                    "makespan": {
                        "mean": float(np.mean(makespans)),
                        "std": float(np.std(makespans)),
                        "best": float(np.min(makespans))
                    },
                    "total_tardiness": {
                        "mean": float(np.mean(tardinesses)),
                        "std": float(np.std(tardinesses)),
                        "best": float(np.min(tardinesses))
                    },
                    "setup_cost": {
                        "mean": float(np.mean(setup_costs)),
                        "std": float(np.std(setup_costs)),
                        "best": float(np.min(setup_costs))
                    },
                    "setup_time": {
                        "mean": float(np.mean(setup_times)),
                        "std": float(np.std(setup_times)),
                        "best": float(np.min(setup_times))
                    },
                    "duration_seconds": {
                        "mean": float(np.mean(durations)),
                        "std": float(np.std(durations))
                    }
                }
            }
            
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4, ensure_ascii=False)
        
    return subfolder_path
