import scipy.io as io
import numpy as np
import os
import hashlib
from typing import Dict, List, Tuple
from src.models import Job, Machine, Workstation, ProblemInstance
from src.config import SMT_PARAMETERS, GA_PARAMETERS


class SMTDataLoader:

    @staticmethod
    def load_mat_problem(file_path: str, seed: int = 42) -> ProblemInstance:
        """
        Loads a MATLAB .mat file containing flow shop scheduling parameters,
        and enriches them with SMT unrelated parallel machine properties.

        Returns:
            ProblemInstance: The loaded and enriched SMT problem instance
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"MATLAB problem file not found at {file_path}")

        data = io.loadmat(file_path)

        # 1. Parse standard dimensions
        num_jobs = int(data["N"].item())
        num_workstations = int(data["M"].item())

        # Unit processing times matrix: shape (M, N) -> TPO[w, j] is unit processing time of job j at workstation w
        tpo = data["TPO"]

        # Setup times matrix: shape (N, N, M) -> s[j, h, w] is setup time at workstation w from job j to job h
        setup_times = data["s"]

        # Due dates: shape (1, N) -> d[0, j] is due date of job j
        due_dates = data["d"].flatten()

        # 2. Initialize deterministic random generator based on seed and file name
        # Hashing the filename ensures that different problems get different random variations even with same seed
        file_hash = (
            int(
                hashlib.md5(os.path.basename(file_path).encode("utf-8")).hexdigest(), 16
            )
            % 10000
        )
        prng = np.random.RandomState(seed + file_hash)

        # 3. Create Workstations and Machines
        workstations = []
        machines_per_ws = SMT_PARAMETERS.Default_Machines_Per_Workstation

        SMT_LCD_STEPS = [
            "SMT Solder Paste Printing",
            "Solder Paste Inspection (SPI)",
            "High-Speed Chip Mounting",
            "Multi-Functional IC Mounting",
            "Reflow Soldering Oven",
            "AOI Optical Inspection",
            "LCD Panel Attachment",
            "FPC Bonding",
            "Final Function Testing",
        ]

        for w in range(num_workstations):
            num_machines = machines_per_ws.get(w, 2)  # Default to 2 if not configured
            ws_name = SMT_LCD_STEPS[w] if w < len(SMT_LCD_STEPS) else f"SMT Stage {w+1}"
            ws = Workstation(id=w, name=ws_name)
            for m in range(num_machines):
                mach = Machine(id=m, workstation_id=w, name=f"W{w+1}_M{m+1}")
                ws.machines.append(mach)
            workstations.append(ws)

        # 4. Generate SMT-specific enriched variables for each job
        jobs = []
        setup_cost_factor = SMT_PARAMETERS.Setup_Cost_Factor
        eligibility_density = SMT_PARAMETERS.Machine_Eligibility_Density

        for j in range(num_jobs):
            # Quantity Q_j in [50, 500] units
            quantity = int(prng.randint(50, 501))

            # Priority group: 1 (highest), 2 (medium), 3 (low), 4 (lowest)
            priority = int(prng.choice([1, 2, 3, 4], p=[0.1, 0.4, 0.3, 0.2]))

            # Material arrival time s_j: uniform in [0, 0.1 * due_date]
            due_date = due_dates[j]
            material_arrival = float(prng.uniform(0, 0.1 * due_date))

            # Unit processing times at workstations
            unit_proc_times = tpo[:, j]

            # Generate Machine Eligibility
            eligible_machines = {}
            for w in range(num_workstations):
                ws = workstations[w]
                num_mach = len(ws.machines)

                # Randomly select eligible machines based on density
                mask = prng.rand(num_mach) < eligibility_density
                eligible_idx = [m for m, active in enumerate(mask) if active]

                # Ensure at least 1 machine is eligible
                if not eligible_idx:
                    eligible_idx = [int(prng.randint(0, num_mach))]

                eligible_machines[w] = eligible_idx

            job = Job(
                id=j,
                quantity=quantity,
                due_date=due_date,
                priority=priority,
                material_arrival_time=material_arrival,
                unit_processing_times=unit_proc_times,
                eligible_machines=eligible_machines,
            )
            jobs.append(job)

        # 5. Generate Setup Costs proportional to setup times
        # setup_cost[j, h, w] = setup_times[j, h, w] * factor
        setup_costs = setup_times * setup_cost_factor

        # 6. Generate transport matrix (initially random, e.g. between 5.0 and 15.0)
        # Shape (num_workstations, num_workstations)
        transport_matrix = prng.uniform(
            5.0, 15.0, size=(num_workstations, num_workstations)
        )
        transport_matrix = np.round(transport_matrix, 1)
        np.fill_diagonal(transport_matrix, 0.0)

        return ProblemInstance(
            jobs=jobs,
            workstations=workstations,
            setup_times=setup_times,
            setup_costs=setup_costs,
            transport_matrix=transport_matrix,
        )
