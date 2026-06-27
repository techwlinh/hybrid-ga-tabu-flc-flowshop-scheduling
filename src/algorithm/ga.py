import numpy as np
import math
import time
from typing import List, Dict, Tuple, Any
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from src.models import Job, Batch, Workstation, ScheduleResult, ProblemInstance
from src.config import GA_PARAMETERS
from src.algorithm.decoder import ChromosomeDecoder
from src.algorithm.fuzzy import FuzzyLogicController
from src.algorithm.tabu import TabuSearch


def _eval_worker(args) -> float:
    """
    Module-level worker function for parallel chromosome evaluation.
    Receives plain dicts/lists instead of dataclass instances to avoid
    pickle class-identity issues caused by Streamlit's module reloading.
    """
    from src.models import Batch, Job, Workstation, Machine

    (
        chromosome,
        batch_dicts,
        job_dicts,
        ws_dicts,
        setup_times,
        setup_costs,
        transport_matrix,
    ) = args

    # Reconstruct Batch objects from plain dicts
    batches = [
        Batch(
            id=b["id"],
            job_id=b["job_id"],
            batch_index=b["batch_index"],
            quantity=b["quantity"],
            processing_times=b["processing_times"],
        )
        for b in batch_dicts
    ]

    # Reconstruct Job objects from plain dicts
    jobs_dict = {}
    for jd in job_dicts:
        jobs_dict[jd["id"]] = Job(
            id=jd["id"],
            quantity=jd["quantity"],
            due_date=jd["due_date"],
            priority=jd["priority"],
            material_arrival_time=jd["material_arrival_time"],
            unit_processing_times=jd["unit_processing_times"],
            eligible_machines=jd["eligible_machines"],
        )

    # Reconstruct Workstation objects from plain dicts
    workstations = [
        Workstation(
            id=wd["id"],
            name=wd["name"],
            machines=[
                Machine(
                    id=md["id"], workstation_id=md["workstation_id"], name=md["name"]
                )
                for md in wd["machines"]
            ],
        )
        for wd in ws_dicts
    ]

    res = ChromosomeDecoder.decode(
        chromosome,
        batches,
        jobs_dict,
        workstations,
        setup_times,
        setup_costs,
        transport_matrix,
    )
    return float(res.total_tardiness + 1e-4 * res.total_setup_cost)


def _serialize_batches(batches):
    """Convert Batch dataclass instances to plain dicts for pickling."""
    return [
        {
            "id": b.id,
            "job_id": b.job_id,
            "batch_index": b.batch_index,
            "quantity": b.quantity,
            "processing_times": b.processing_times,
        }
        for b in batches
    ]


def _serialize_jobs_dict(jobs_dict):
    """Convert Job dataclass instances to plain dicts for pickling."""
    return [
        {
            "id": j.id,
            "quantity": j.quantity,
            "due_date": j.due_date,
            "priority": j.priority,
            "material_arrival_time": j.material_arrival_time,
            "unit_processing_times": j.unit_processing_times,
            "eligible_machines": j.eligible_machines,
        }
        for j in jobs_dict.values()
    ]


def _serialize_workstations(workstations):
    """Convert Workstation/Machine dataclass instances to plain dicts for pickling."""
    return [
        {
            "id": ws.id,
            "name": ws.name,
            "machines": [
                {"id": m.id, "workstation_id": m.workstation_id, "name": m.name}
                for m in ws.machines
            ],
        }
        for ws in workstations
    ]


class HFGA_TS:
    """
    Hybrid Fuzzy Genetic Algorithm with Tabu Search (HFGA-TS)
    for optimizing SMT scheduling.
    """

    def __init__(self, problem: ProblemInstance, use_flc: bool = True, use_tabu: bool = True):
        self.problem = problem
        self.jobs_dict = {job.id: job for job in problem.jobs}
        self.use_flc = use_flc
        self.use_tabu = use_tabu

        # Load parameters
        self.splitting_threshold = GA_PARAMETERS.Threshold_of_Batch_Splitting
        self.pop_size = GA_PARAMETERS.Population_Size
        self.max_generations = GA_PARAMETERS.Maximum_Generation
        self.elitism_rate = GA_PARAMETERS.Elitism_Rate
        self.stall_limit = GA_PARAMETERS.No_Improvement_Limit
        self.time_limit = GA_PARAMETERS.Time_Limitation_Seconds
        self.use_parallel = GA_PARAMETERS.Use_Parallel_Execution

        # Initialize sub-modules
        self.flc = FuzzyLogicController()
        self.ts = TabuSearch()

        # 1. Preprocess: Split jobs into batches
        self.batches = self._batch_splitting_preprocessing(problem.jobs)
        self.N_batches = len(self.batches)

    def _batch_splitting_preprocessing(self, jobs: List[Job]) -> List[Batch]:
        """
        Splits large job quantities into parallel-run batches if their average
        total processing time exceeds the threshold T.
        """
        batches = []
        num_workstations = len(self.problem.workstations)

        for job in jobs:
            # Calculate the nominal/average total processing time for the job quantity
            # nominal_pt = Average total processing time across workstations
            total_unit_pt = sum(job.unit_processing_times)
            nominal_pt = (total_unit_pt * job.quantity) / num_workstations

            if nominal_pt > self.splitting_threshold:
                # Calculate number of batches b_j
                b_j = math.ceil(nominal_pt / self.splitting_threshold)

                # Distribute quantity Q_j among b_j batches
                base_qty = job.quantity // b_j
                remainder = job.quantity % b_j

                for k in range(b_j):
                    # Add remainder to the first few batches
                    batch_qty = base_qty + (1 if k < remainder else 0)
                    if batch_qty <= 0:
                        continue

                    batch_id = f"J{job.id}_B{k}"
                    # Scaled processing times at each stage for this batch quantity
                    proc_times = job.unit_processing_times * batch_qty

                    batch = Batch(
                        id=batch_id,
                        job_id=job.id,
                        batch_index=k,
                        quantity=batch_qty,
                        processing_times=proc_times,
                    )
                    batches.append(batch)
            else:
                # No splitting required
                batch = Batch(
                    id=f"J{job.id}_B0",
                    job_id=job.id,
                    batch_index=0,
                    quantity=job.quantity,
                    processing_times=job.unit_processing_times * job.quantity,
                )
                batches.append(batch)

        return batches

    @staticmethod
    def _evaluate_fitness(res: ScheduleResult) -> float:
        """
        Calculates a composite scalar fitness score to minimize.
        Primary objective: Total Tardiness.
        Secondary objective: Setup Cost (weighted very small to resolve ties).
        """
        return float(res.total_tardiness + 1e-4 * res.total_setup_cost)

    def _evaluate_population(
        self, population: np.ndarray, executor: ProcessPoolExecutor = None
    ) -> np.ndarray:
        """Evaluates fitnesses of a population of chromosomes, using multiprocessing if enabled."""
        pop_size = len(population)
        transport_mat = getattr(self.problem, "transport_matrix", None)
        if self.use_parallel and executor is not None:
            # Serialize dataclass objects to plain dicts to avoid pickle class-identity
            # issues when running under Streamlit's module reloading
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

    def _select_parent(
        self, population: np.ndarray, fitnesses: np.ndarray
    ) -> np.ndarray:
        """Tournament selection of size 3."""
        idx = np.random.choice(self.pop_size, size=3, replace=False)
        best_idx = idx[np.argmin(fitnesses[idx])]
        return population[best_idx].copy()

    def _crossover(
        self, p1: np.ndarray, p2: np.ndarray, pc: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Uniform Crossover with probability Pc."""
        if np.random.rand() > pc:
            return p1.copy(), p2.copy()

        c1, c2 = p1.copy(), p2.copy()
        mask = np.random.rand(self.N_batches * 2) < 0.5
        c1[mask] = p2[mask]
        c2[mask] = p1[mask]
        return c1, c2

    def _mutate(self, chrom: np.ndarray, pm: float) -> np.ndarray:
        """Random reset mutation: replaces genes with random [0, 1] with probability Pm."""
        mutated = chrom.copy()
        mask = np.random.rand(self.N_batches * 2) < pm
        mutated[mask] = np.random.rand(np.sum(mask))
        return mutated

    def run(
        self, progress_callback: Any = None
    ) -> Tuple[ScheduleResult, np.ndarray, Dict[str, Any]]:
        """
        Runs the full HFGA-TS optimization loop.

        Args:
            progress_callback: A function taking (gen, best_fit, avg_fit, div, pc, pm) for Streamlit live charts.
        """
        start_time = time.time()

        # 1. Initialize random key population
        # Each chromosome has length 2 * N_batches
        population = np.random.rand(self.pop_size, self.N_batches * 2)
        fitnesses = np.zeros(self.pop_size)

        # History logger
        history = {
            "generation": [],
            "best_fitness": [],
            "average_fitness": [],
            "diversity": [],
            "pc": [],
            "pm": [],
        }

        # Initialize parallel executor if enabled
        executor = None
        if self.use_parallel:
            num_workers = max(1, multiprocessing.cpu_count() - 1)
            executor = ProcessPoolExecutor(max_workers=num_workers)

        try:
            # Initial evaluation
            fitnesses = self._evaluate_population(population, executor)

            best_idx = np.argmin(fitnesses)
            best_chrom = population[best_idx].copy()
            best_fit = fitnesses[best_idx]

            # Stall and FLC trackers
            stall_counter = 0
            pc = GA_PARAMETERS.Initial_Crossover_Rate
            pm = GA_PARAMETERS.Initial_Mutation_Rate

            # Track historical average fitnesses for FLC improvement input
            prev_avg_fit = np.mean(fitnesses)

            # 2. Main Generations Loop
            for gen in range(self.max_generations):
                # Check runtime limit
                if time.time() - start_time > self.time_limit:
                    break

                curr_avg_fit = np.mean(fitnesses)

                # Compute Population Diversity (Coefficient of Variation)
                std_fit = np.std(fitnesses)
                diversity = std_fit / (curr_avg_fit + 1e-6)
                # Normalize diversity to [0, 1] relative to a sensible upper limit (e.g. CV=0.5 is very diverse)
                norm_diversity = min(diversity / 0.5, 1.0)

                # Compute Improvement Speed (Input 2 of FLC)
                improvement = (curr_avg_fit - prev_avg_fit) / (best_fit + 1e-6)
                prev_avg_fit = curr_avg_fit

                # FLC adjustment of Pc and Pm
                if self.use_flc:
                    pc, pm = self.flc.evaluate(norm_diversity, improvement)
                else:
                    pc = GA_PARAMETERS.Initial_Crossover_Rate
                    pm = GA_PARAMETERS.Initial_Mutation_Rate

                # Elitism: retain top performers
                num_elites = int(self.pop_size * self.elitism_rate)
                elites_idx = np.argsort(fitnesses)[:num_elites]
                new_population = population[elites_idx].copy()
                new_fitnesses = fitnesses[elites_idx].copy()

                # Recreate rest of population
                child_slots = self.pop_size - num_elites
                children = []

                while len(children) < child_slots:
                    p1 = self._select_parent(population, fitnesses)
                    p2 = self._select_parent(population, fitnesses)

                    c1, c2 = self._crossover(p1, p2, pc)
                    c1 = self._mutate(c1, pm)
                    c2 = self._mutate(c2, pm)

                    children.append(c1)
                    if len(children) < child_slots:
                        children.append(c2)

                children = np.array(children)

                # Evaluate children (using multiprocessing if enabled)
                child_fitnesses = self._evaluate_population(children, executor)

                # Combine into new population
                population = np.vstack([new_population, children])
                fitnesses = np.concatenate([new_fitnesses, child_fitnesses])

                # Find best in new population
                gen_best_idx = np.argmin(fitnesses)
                gen_best_fit = fitnesses[gen_best_idx]

                if gen_best_fit < best_fit:
                    best_fit = gen_best_fit
                    best_chrom = population[gen_best_idx].copy()
                    stall_counter = 0
                else:
                    stall_counter += 1

                # 3. Tabu Search activation if evolution stalls
                if stall_counter >= self.stall_limit:
                    if self.use_tabu:
                        # Run Tabu Search to optimize the current best chromosome
                        optimized_chrom = self.ts.optimize(
                            best_chrom,
                            self.batches,
                            self.jobs_dict,
                            self.problem.workstations,
                            self.problem.setup_times,
                            self.problem.setup_costs,
                            ChromosomeDecoder.decode,
                            self._evaluate_fitness,
                            getattr(self.problem, "transport_matrix", None),
                        )

                        opt_res = ChromosomeDecoder.decode(
                            optimized_chrom,
                            self.batches,
                            self.jobs_dict,
                            self.problem.workstations,
                            self.problem.setup_times,
                            self.problem.setup_costs,
                            getattr(self.problem, "transport_matrix", None),
                        )

                        opt_fit = self._evaluate_fitness(opt_res)

                        if opt_fit < best_fit:
                            best_fit = opt_fit
                            best_chrom = optimized_chrom.copy()

                            # Inject back into population (replace the worst individual)
                            worst_idx = np.argmax(fitnesses)
                            population[worst_idx] = optimized_chrom.copy()
                            fitnesses[worst_idx] = opt_fit

                    stall_counter = 0  # Reset counter after local search

                # Log history
                history["generation"].append(gen + 1)
                history["best_fitness"].append(best_fit)
                history["average_fitness"].append(curr_avg_fit)
                history["diversity"].append(norm_diversity)
                history["pc"].append(pc)
                history["pm"].append(pm)

                if progress_callback is not None:
                    progress_callback(
                        generation=gen + 1,
                        best_fitness=best_fit,
                        average_fitness=curr_avg_fit,
                        diversity=norm_diversity,
                        pc=pc,
                        pm=pm,
                    )
        finally:
            # Gracefully clean up workers
            if executor is not None:
                executor.shutdown()

        # Final decode of the best chromosome to get full schedule
        best_schedule_result = ChromosomeDecoder.decode(
            best_chrom,
            self.batches,
            self.jobs_dict,
            self.problem.workstations,
            self.problem.setup_times,
            self.problem.setup_costs,
            getattr(self.problem, "transport_matrix", None),
        )

        return best_schedule_result, best_chrom, history
