import numpy as np
import copy
from typing import List, Tuple, Callable, Any, Dict
from src.config import GA_PARAMETERS

class TabuSearch:
    """
    Tabu Search local search optimizer that refines the job sequence section
    of the best Genetic Algorithm chromosomes to escape local optima.
    """
    def __init__(self):
        self.max_iter = GA_PARAMETERS["Max_Iterations_of_Tabu_Search"]
        self.tabu_size = GA_PARAMETERS["Tabu_List_Size"]

    def optimize(
        self,
        initial_chromosome: np.ndarray,
        batches: List[Any],
        jobs_dict: Dict[int, Any],
        workstations: List[Any],
        setup_times: np.ndarray,
        setup_costs: np.ndarray,
        decode_fn: Callable,
        evaluate_fn: Callable[[Any], float],
        transport_matrix: np.ndarray = None
    ) -> np.ndarray:
        """
        Performs local search on the sequence genes (first half of the chromosome).
        
        Args:
            initial_chromosome: 1D numpy array of size 2 * N_batches
            batches: list of Batch objects
            workstations: list of Workstation objects
            setup_times: setup times matrix
            setup_costs: setup costs matrix
            decode_fn: function mapping chromosome to ScheduleResult
            evaluate_fn: function mapping ScheduleResult to a scalar fitness value (to minimize)
            transport_matrix: optional transport times matrix between workstations
            
        Returns:
            best_chromosome: Enriched/optimized chromosome
        """
        N_batches = len(batches)
        if N_batches < 2:
            return initial_chromosome # Nothing to swap/insert
            
        # Copy chromosome
        curr_chrom = initial_chromosome.copy()
        curr_res = decode_fn(curr_chrom, batches, jobs_dict, workstations, setup_times, setup_costs, transport_matrix)
        curr_fit = evaluate_fn(curr_res)
        
        best_chrom = curr_chrom.copy()
        best_fit = curr_fit
        
        # Tabu list stores elements in FIFO queue
        # For swap: ("swap", idx1, idx2)
        # For insert: ("insert", idx1, idx2)
        tabu_list = []
        
        # Seed generator
        prng = np.random.RandomState(42)
        
        for iteration in range(self.max_iter):
            candidates = []
            
            # Generate neighborhood candidates (e.g. 20 candidates)
            num_candidates = min(20, N_batches * (N_batches - 1))
            
            for _ in range(num_candidates):
                candidate_chrom = curr_chrom.copy()
                move_type = prng.choice(["swap", "insert"])
                
                idx1, idx2 = prng.choice(N_batches, size=2, replace=False)
                
                # Apply move on the first half (sequence part)
                if move_type == "swap":
                    # Swap values
                    candidate_chrom[idx1], candidate_chrom[idx2] = candidate_chrom[idx2], candidate_chrom[idx1]
                    move_repr = ("swap", min(idx1, idx2), max(idx1, idx2))
                else: # insert
                    # Remove from idx1 and insert at idx2
                    gene = candidate_chrom[idx1]
                    seq_part = list(candidate_chrom[:N_batches])
                    seq_part.pop(idx1)
                    seq_part.insert(idx2, gene)
                    # Put back in candidate
                    candidate_chrom[:N_batches] = np.array(seq_part)
                    move_repr = ("insert", idx1, idx2)
                    
                # Decode and evaluate
                res = decode_fn(candidate_chrom, batches, jobs_dict, workstations, setup_times, setup_costs, transport_matrix)
                fit = evaluate_fn(res)

                
                candidates.append((candidate_chrom, fit, move_repr))
                
            # Sort candidates by fitness (ascending - we want to minimize)
            candidates.sort(key=lambda x: x[1])
            
            # Find the best valid candidate
            selected_candidate = None
            for cand_chrom, cand_fit, cand_move in candidates:
                is_tabu = cand_move in tabu_list
                
                # Aspiration criterion: if the tabu move yields a better result than global best
                if not is_tabu or (cand_fit < best_fit):
                    selected_candidate = (cand_chrom, cand_fit, cand_move)
                    break
                    
            if selected_candidate is None:
                # If all moves are tabu and none pass aspiration, pick the absolute best candidate
                if candidates:
                    selected_candidate = candidates[0]
                else:
                    break
                    
            cand_chrom, cand_fit, cand_move = selected_candidate
            
            # Update current state
            curr_chrom = cand_chrom
            curr_fit = cand_fit
            
            # Add to tabu list
            tabu_list.append(cand_move)
            if len(tabu_list) > self.tabu_size:
                tabu_list.pop(0) # FIFO pop
                
            # Update global best
            if cand_fit < best_fit:
                best_chrom = cand_chrom.copy()
                best_fit = cand_fit
                
        return best_chrom
