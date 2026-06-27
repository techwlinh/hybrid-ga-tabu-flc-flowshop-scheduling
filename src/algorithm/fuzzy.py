import numpy as np
from typing import Tuple
from src.config import GA_PARAMETERS

class FuzzyLogicController:
    """
    Pure Python Mamdani Fuzzy Logic Controller to dynamically adjust
    crossover rate (Pc) and mutation rate (Pm) at each generation.
    """
    def __init__(self):
        # 1. Define bounds for input and output variables
        self.pc_bounds = GA_PARAMETERS.Crossover_Rate_Bounds
        self.pm_bounds = GA_PARAMETERS.Mutation_Rate_Bounds
        
        # 2. Define the membership function parameters (Triangular [a, b, c])
        # Input 1: Diversity (normalized to [0, 1])
        self.div_sets = {
            "NL": [-0.25, 0.0, 0.25],
            "NS": [0.0, 0.25, 0.5],
            "ZE": [0.25, 0.5, 0.75],
            "PS": [0.5, 0.75, 1.0],
            "PL": [0.75, 1.0, 1.25]
        }
        
        # Input 2: Improvement Speed (range [-0.5, 0.5])
        self.imp_sets = {
            "NL": [-0.5, -0.5, -0.25],
            "NS": [-0.5, -0.25, 0.0],
            "ZE": [-0.25, 0.0, 0.25],
            "PS": [0.0, 0.25, 0.5],
            "PL": [0.25, 0.5, 0.5]
        }
        
        # Output 1: Crossover Rate (Pc in [0.5, 0.95])
        pc_min, pc_max = self.pc_bounds
        pc_mid = (pc_min + pc_max) / 2
        pc_q1 = pc_min + (pc_max - pc_min) / 4
        pc_q3 = pc_min + 3 * (pc_max - pc_min) / 4
        
        self.pc_sets = {
            "NL": [pc_min, pc_min, pc_q1],
            "NS": [pc_min, pc_q1, pc_mid],
            "ZE": [pc_q1, pc_mid, pc_q3],
            "PS": [pc_mid, pc_q3, pc_max],
            "PL": [pc_q3, pc_max, pc_max]
        }
        
        # Output 2: Mutation Rate (Pm in [0.05, 0.2])
        pm_min, pm_max = self.pm_bounds
        pm_mid = (pm_min + pm_max) / 2
        pm_q1 = pm_min + (pm_max - pm_min) / 4
        pm_q3 = pm_min + 3 * (pm_max - pm_min) / 4
        
        self.pm_sets = {
            "NL": [pm_min, pm_min, pm_q1],
            "NS": [pm_min, pm_q1, pm_mid],
            "ZE": [pm_q1, pm_mid, pm_q3],
            "PS": [pm_mid, pm_q3, pm_max],
            "PL": [pm_q3, pm_max, pm_max]
        }
        
        # 3. Define Rule Base: rule_matrix[Div][Imp] -> Output Fuzzy Set Key
        # Mapping for Pc
        self.rules_pc = {
            "NL": {"NL": "PL", "NS": "PS", "ZE": "NS", "PS": "NL", "PL": "NL"},
            "NS": {"NL": "PL", "NS": "PS", "ZE": "ZE", "PS": "NS", "PL": "NL"},
            "ZE": {"NL": "PL", "NS": "ZE", "ZE": "ZE", "PS": "NS", "PL": "NS"},
            "PS": {"NL": "PS", "NS": "ZE", "ZE": "ZE", "PS": "ZE", "PL": "ZE"},
            "PL": {"NL": "PS", "NS": "PS", "ZE": "ZE", "PS": "ZE", "PL": "ZE"}
        }
        
        # Mapping for Pm
        self.rules_pm = {
            "NL": {"NL": "NL", "NS": "NS", "ZE": "PL", "PS": "PL", "PL": "PL"},
            "NS": {"NL": "NL", "NS": "NS", "ZE": "PS", "PS": "PL", "PL": "PL"},
            "ZE": {"NL": "NS", "NS": "ZE", "ZE": "ZE", "PS": "PS", "PL": "PS"},
            "PS": {"NL": "ZE", "NS": "ZE", "ZE": "NS", "PS": "NS", "PL": "ZE"},
            "PL": {"NL": "ZE", "NS": "ZE", "ZE": "NL", "PS": "NL", "PL": "NL"}
        }

    def _trimf(self, x: float, abc: list) -> float:
        """Evaluates a triangular membership function at point x."""
        a, b, c = abc
        if x <= a or x >= c:
            return 0.0
        elif a < x <= b:
            if a == b:
                return 1.0
            return (x - a) / (b - a)
        else:
            if b == c:
                return 1.0
            return (c - x) / (c - b)

    def evaluate(self, diversity: float, improvement: float) -> Tuple[float, float]:
        """
        Runs Mamdani fuzzy inference on crisp inputs (diversity, improvement)
        and returns the defuzzified crisp values for (Pc, Pm).
        """
        # Clip inputs to active domains
        div_val = float(np.clip(diversity, 0.0, 1.0))
        imp_val = float(np.clip(improvement, -0.5, 0.5))
        
        # 1. Fuzzification
        mu_div = {k: self._trimf(div_val, params) for k, params in self.div_sets.items()}
        mu_imp = {k: self._trimf(imp_val, params) for k, params in self.imp_sets.items()}
        
        # 2. Rule evaluation and aggregation
        # Discretize outputs to evaluate the aggregated shape
        grid_points = 50
        pc_grid = np.linspace(self.pc_bounds[0], self.pc_bounds[1], grid_points)
        pm_grid = np.linspace(self.pm_bounds[0], self.pm_bounds[1], grid_points)
        
        pc_aggregated = np.zeros(grid_points)
        pm_aggregated = np.zeros(grid_points)
        
        # Iterate over all combinations (rules)
        for div_key, div_memb in mu_div.items():
            if div_memb <= 0:
                continue
            for imp_key, imp_memb in mu_imp.items():
                if imp_memb <= 0:
                    continue
                    
                # Rule activation strength (AND operator is min)
                w = min(div_memb, imp_memb)
                
                # Retrieve matching output sets
                pc_out_key = self.rules_pc[div_key][imp_key]
                pm_out_key = self.rules_pm[div_key][imp_key]
                
                # Aggregate for Pc
                pc_params = self.pc_sets[pc_out_key]
                for idx, y in enumerate(pc_grid):
                    val = min(w, self._trimf(y, pc_params))
                    pc_aggregated[idx] = max(pc_aggregated[idx], val)
                    
                # Aggregate for Pm
                pm_params = self.pm_sets[pm_out_key]
                for idx, z in enumerate(pm_grid):
                    val = min(w, self._trimf(z, pm_params))
                    pm_aggregated[idx] = max(pm_aggregated[idx], val)
                    
        # 3. Defuzzification (Centroid Method)
        sum_pc_memb = np.sum(pc_aggregated)
        if sum_pc_memb > 0:
            pc_final = np.sum(pc_grid * pc_aggregated) / sum_pc_memb
        else:
            pc_final = (self.pc_bounds[0] + self.pc_bounds[1]) / 2.0
            
        sum_pm_memb = np.sum(pm_aggregated)
        if sum_pm_memb > 0:
            pm_final = np.sum(pm_grid * pm_aggregated) / sum_pm_memb
        else:
            pm_final = (self.pm_bounds[0] + self.pm_bounds[1]) / 2.0
            
        return float(pc_final), float(pm_final)
