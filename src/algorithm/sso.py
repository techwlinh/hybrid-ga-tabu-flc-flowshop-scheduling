import numpy as np
import time
from typing import List, Dict, Tuple, Any
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from src.models import Batch, Job, Workstation, Machine, ScheduleResult, ProblemInstance
from src.config import GA_PARAMETERS
from src.algorithm.decoder import ChromosomeDecoder

class SSOScheduler:
    """
    Simplified Swarm Optimization (SSO) SMT scheduling optimizer.
    Uses the exact same continuous chromosome representation as GA,
    allowing direct comparison.
    """
    def __init__(self, problem: ProblemInstance):
        self.problem = problem
        self.jobs_dict = {job.id: job for job in problem.jobs}
        
        # Split jobs into batches using the GA's batch splitting logic to ensure comparability
        from src.algorithm.ga import HFGA_TS
        temp_ga = HFGA_TS(problem)
        self.batches = temp_ga.batches
        self.N_batches = len(self.batches)
        
        self.pop_size = GA_PARAMETERS.Population_Size
        self.max_generations = GA_PARAMETERS.Maximum_Generation
        self.use_parallel = GA_PARAMETERS.Use_Parallel_Execution
        self.time_limit = GA_PARAMETERS.Time_Limitation_Seconds
        
        # SSO parameters (probabilities Cw + Cp + Cg + Cr = 1)
        cw = getattr(GA_PARAMETERS, "SSO_Cw", 0.20)
        cp = getattr(GA_PARAMETERS, "SSO_Cp", 0.20)
        cg = getattr(GA_PARAMETERS, "SSO_Cg", 0.50)
        cr = getattr(GA_PARAMETERS, "SSO_Cr", 0.10)
        
        tot = cw + cp + cg + cr
        if tot > 0:
            self.Cw = cw / tot
            self.Cp = cp / tot
            self.Cg = cg / tot
            self.Cr = cr / tot
        else:
            self.Cw, self.Cp, self.Cg, self.Cr = 0.20, 0.20, 0.50, 0.10

    def _evaluate_fitness(self, res: ScheduleResult) -> float:
        """
        Weighted combination of Total Tardiness and Makespan.
        """
        from src.config import GA_PARAMETERS
        alpha = getattr(GA_PARAMETERS, "fitness_alpha", 0.5)
        beta = getattr(GA_PARAMETERS, "fitness_beta", 0.5)
        return float(alpha * res.total_tardiness + beta * res.makespan)

    def _evaluate_population(
        self, population: np.ndarray, executor: ProcessPoolExecutor = None
    ) -> np.ndarray:
        """Evaluates a population/swarm using parallel workers if enabled."""
        from src.algorithm.ga import (
            _serialize_batches,
            _serialize_jobs_dict,
            _serialize_workstations,
            _eval_worker,
        )
        pop_size = len(population)
        transport_mat = getattr(self.problem, "transport_matrix", None)
        
        if self.use_parallel and executor is not None:
            batch_dicts = _serialize_batches(self.batches)
            job_dicts = _serialize_jobs_dict(self.jobs_dict)
            ws_dicts = _serialize_workstations(self.problem.workstations)
            tasks_args = [
                (
                    population[i],
                    batch_dicts,
                    job_dicts,
                    ws_dicts,
                    self.problem.setup_times,
                    self.problem.setup_costs,
                    transport_mat,
                )
                for i in range(pop_size)
            ]
            fit_list = list(executor.map(_eval_worker, tasks_args))
            return np.array(fit_list)
        else:
            fitnesses = np.zeros(pop_size)
            for i in range(pop_size):
                res = ChromosomeDecoder.decode(
                    population[i],
                    self.batches,
                    self.jobs_dict,
                    self.problem.workstations,
                    self.problem.setup_times,
                    self.problem.setup_costs,
                    transport_mat,
                )
                fitnesses[i] = self._evaluate_fitness(res)
            return fitnesses

    def run(self, progress_callback: Any = None) -> Tuple[ScheduleResult, np.ndarray, Dict[str, Any]]:
        """
        Runs the SSO Swarm optimization loop.
        """
        start_time = time.time()
        
        # 1. Initialize random swarm
        population = np.random.rand(self.pop_size, self.N_batches * 2)
        fitnesses = np.zeros(self.pop_size)
        
        # Personal bests
        pbest_pos = population.copy()
        pbest_fit = np.full(self.pop_size, float("inf"))
        
        history = {
            "generation": [],
            "best_fitness": [],
            "average_fitness": [],
        }

        # Multiprocessing executor
        executor = None
        if self.use_parallel:
            num_workers = max(1, multiprocessing.cpu_count() - 1)
            executor = ProcessPoolExecutor(max_workers=num_workers)

        try:
            # Initial evaluation
            fitnesses = self._evaluate_population(population, executor)
            pbest_fit = fitnesses.copy()
            
            best_idx = np.argmin(fitnesses)
            gbest_pos = population[best_idx].copy()
            gbest_fit = fitnesses[best_idx]
            
            # 2. Swarm Evolution Loop
            for gen in range(self.max_generations):
                if time.time() - start_time > self.time_limit:
                    break
                
                curr_avg_fit = np.mean(fitnesses)
                
                # Update each particle
                for i in range(self.pop_size):
                    rands = np.random.rand(self.N_batches * 2)
                    for j in range(self.N_batches * 2):
                        r = rands[j]
                        if r < self.Cw:
                            # Keep current position
                            pass
                        elif r < self.Cw + self.Cp:
                            # Set to personal best
                            population[i, j] = pbest_pos[i, j]
                        elif r < self.Cw + self.Cp + self.Cg:
                            # Set to global best
                            population[i, j] = gbest_pos[j]
                        else:
                            # Exploration (random value in [0, 1])
                            population[i, j] = np.random.rand()

                # Re-evaluate
                fitnesses = self._evaluate_population(population, executor)

                # Update personal bests
                for i in range(self.pop_size):
                    if fitnesses[i] < pbest_fit[i]:
                        pbest_fit[i] = fitnesses[i]
                        pbest_pos[i] = population[i].copy()

                # Update global best
                gen_best_idx = np.argmin(fitnesses)
                if fitnesses[gen_best_idx] < gbest_fit:
                    gbest_fit = fitnesses[gen_best_idx]
                    gbest_pos = population[gen_best_idx].copy()

                history["generation"].append(gen + 1)
                history["best_fitness"].append(gbest_fit)
                history["average_fitness"].append(curr_avg_fit)

                if progress_callback is not None:
                    # Provide FLC variables as 0 for SSO
                    progress_callback(
                        generation=gen + 1,
                        best_fitness=gbest_fit,
                        average_fitness=curr_avg_fit,
                        diversity=0.0,
                        pc=0.0,
                        pm=0.0,
                    )
        finally:
            if executor is not None:
                executor.shutdown()

        # Decode best global particle
        best_schedule_result = ChromosomeDecoder.decode(
            gbest_pos,
            self.batches,
            self.jobs_dict,
            self.problem.workstations,
            self.problem.setup_times,
            self.problem.setup_costs,
            getattr(self.problem, "transport_matrix", None),
        )

        return best_schedule_result, gbest_pos, history
