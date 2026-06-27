# Implementation Plan - SMT Scheduling System (HFGA-TS)

This plan details the design and implementation of the **Hybrid Fuzzy Genetic Algorithm with Tabu Search (HFGA-TS)** for SMT scheduling, utilizing the dataset from Antonella Branda (2021) and enriching it to meet SMT constraints.

## Goal Description
Build a professional, explainable optimization system for SMT line scheduling in a **Hybrid Flow Shop (HFS)** setup, complete with a command-line runner and an interactive **Streamlit web application** for testing, hyperparameter tuning, and schedule analysis.

---

## SMT Domain Model & Data Mapping

To match real-world SMT assembly lines, the dataset from `.mat` files will be adapted and enriched as follows:

1. **Workstations & Machines:**
   - The $M$ stages in the `.mat` file represent **Workstations** (e.g., Printing, SPI, Placement 1, Placement 2, Reflow).
   - Each Workstation $w$ contains a set of $m_w$ parallel identical machines (except for dedicated machines). The number of machines per workstation is configurable (default is 2 or 3).
2. **Job Quantities & Processing Times:**
   - Instead of a fixed processing time, each job $j$ has a **quantity of units** $Q_j$ (e.g., between 50 and 500 units, generated randomly).
   - The cell `TPO[w, j]` represents the **unit processing time** of job $j$ at workstation $w$.
   - The total processing time of job $j$ at workstation $w$ is $P_{wj} = TPO[w, j] \times Q_j$.
3. **Batch Splitting:**
   - If the total nominal processing time of job $j$ exceeds $T_{\text{threshold}}$, it is split into $b_j$ batches.
   - The quantity $Q_j$ is distributed among these $b_j$ batches (e.g., $q_{jk} = Q_j / b_j$).
   - The processing time of batch $k$ at workstation $w$ is $P_{w, j, k} = TPO[w, j] \times q_{jk}$.
4. **Transport Time:**
   - A transit time $TT_w$ is introduced between workstation $w$ and $w+1$ (default is 5.0 time units). A batch must wait for this transport time after finishing at workstation $w$ before it can start setup/processing at workstation $w+1$.
5. **Setup Costs:**
   - A sequence-dependent setup cost matrix `setup_cost[j, h, w]` is introduced at each workstation, proportional to the setup time `s[j, h, w]`.
6. **Other Constraints:**
   - **Priority ($pr_j$):** Randomly assigned to levels $\{1, 2, 3\}$ (1 is highest priority).
   - **Material Arrival ($s_j$):** Randomly generated in the range $[0, 0.1 \times D_j]$.
   - **Machine Eligibility ($E_{ij}$):** Binary eligibility matrix. Some machines at a workstation can be marked as ineligible for certain jobs.

---

## User Review Required

> [!IMPORTANT]
> **Streamlit Testing Playground & Interactive Gantt**
> The Streamlit application will allow users to:
> 1. Select any `.mat` dataset file from the `Problems` directory and view its metadata (jobs, workstations, processing/setup ranges).
> 2. Adjust GA parameters, FLC bounds, Tabu Search limits, and SMT constraints (machine counts, transport times).
> 3. Run the optimization and view real-time charts (fitness improvement, diversity, FLC rate changes).
> 4. Analyze the optimized schedule with an **interactive Plotly Gantt chart**:
>    - **Workstation-based grouping** on the y-axis, with expandable sub-rows for parallel machines.
>    - **Zoom/Pan/Filter** controls.
>    - **Detailed tooltips** showing Job ID, Batch ID, quantity, start/end times, priority, due date, tardiness, setup time/cost, and transit delays on hover.
>    - **Highlighting** connected batches of the same job across different workstations.

> [!TIP]
> **Explainable Optimization & Solution Analysis**
> We will implement a "Solution Explainer" component. After running the optimization, the tool will analyze the solution and print/display natural language explanations:
> - Identify the bottleneck workstation (the workstation with the highest queue/utilization).
> - List critical tardy jobs and explain *why* they were delayed (e.g., late material arrival, low priority, high changeover overhead).
> - Explain specific scheduling decisions, such as why a batch was split and how it helped balance parallel machines at a workstation.

---

## Proposed Directory & Code Structure

We will implement a professional, modular structure to keep files small, readable, and easy to maintain.

```
flowchart/
  └── system_architecture.drawio # [NEW] XML system architecture, ERD, and algorithm flowchart
src/
  ├── __init__.py
  ├── config.py             # Global configurations and SMT default parameters
  ├── models.py             # Data classes (Job, Batch, Workstation, Machine, ScheduleEntry, ScheduleResult)
  ├── data/
  │     ├── __init__.py
  │     └── loader.py       # Scipy .mat loading, SMT quantity scaling & parameter enrichment
  ├── algorithm/
  │     ├── __init__.py
  │     ├── decoder.py      # Chromosome decoder (HFS routing, setup times, transport delays)
  │     ├── fuzzy.py        # Mamdani Fuzzy Logic Controller (Pc and Pm adjustment)
  │     ├── tabu.py         # Tabu Search neighborhood optimizer
  │     └── ga.py           # Main Genetic Algorithm loop with elitism and FLC/TS hooks
  ├── visualization/
  │     ├── __init__.py
  │     └── gantt.py        # Plotly interactive Gantt & Matplotlib fallback
  └── app.py                # Streamlit dashboard and optimization playground
```

---

## Component Details

### `flowchart/system_architecture.drawio`
An XML-based diagram detailing the system's structural blueprint:
- **SMT Domain Data Model & ERD:** Outlines the tables (Job, Batch, Workstation, Machine, Setup Matrix, Transport Time, Schedule Result) and their relationships.
- **Algorithm Flow Logic (HFGA-TS):** Outlines the optimization sequence from preprocessing (batch splitting), random key representation, HFS decoding, fitness evaluation, Mamdani FLC adjustments, to Tabu Search local search.
- **Streamlit App Workspace & Flow:** Visualizes the dashboard's data processing pipeline and visualization panels.

### `src/config.py`
Defines default SMT parameters:
- `DEFAULT_MACHINES_PER_WORKSTATION = {0: 2, 1: 2, 2: 3, 3: 2, 4: 2}` (or auto-generated based on workstation count)
- `DEFAULT_TRANSPORT_TIME = 10.0`
- `SETUP_COST_FACTOR = 1.5`
- Default GA/FLC/TS parameters matching Yang Zih-Yueh's paper.

### `src/models.py`
Uses Python dataclasses or Pydantic to represent problem entities:
- `Job`: `id`, `quantity`, `due_date`, `priority`, `material_arrival_time`, `unit_processing_times` (array of length M), `eligible_machines` (dict workstation_id -> list of machine_ids).
- `Batch`: `id`, `job_id`, `batch_index`, `quantity`, `processing_times` (array of length M).
- `ScheduleEntry`: `batch_id`, `job_id`, `workstation_id`, `machine_id`, `start_time`, `end_time`, `setup_time`, `setup_cost`, `waiting_time`.

### `src/data/loader.py`
Extracts raw matrices from `.mat` and enriches them:
- Generates job quantities $Q_j \in [50, 500]$ and priorities.
- Computes setup costs as `setup_time * SETUP_COST_FACTOR + noise`.
- Enforces reproducible randomness by hashing the file content or seeding with the problem ID.

### `src/algorithm/decoder.py`
A crucial module mapping the chromosome `[Seq_Genes | Mach_Genes]` to a valid HFS schedule:
- Sorts batches by `(priority, seq_gene)`.
- For each batch, routes it sequentially through workstations $0 \dots M-1$:
  - Finds the machine at workstation $w$ assigned by the `mach_gene`.
  - Factors in the machine's free clock, setup time from the previous job on that machine, and transport time from the previous stage.

### `src/algorithm/fuzzy.py`
Pure-Python Mamdani FLC that maps population diversity and improvement rate to crossover and mutation rates.
- Normalizes population diversity via coefficient of variation.
- Evaluates rules and applies centroid defuzzification using a discretized grid.

### `src/algorithm/tabu.py`
Applies Tabu Search local search to the top chromosome.
- Swap and insert moves on the sequence section of the chromosome.
- Restores and respects the best chromosome using aspiration criteria.

### `src/app.py`
A Streamlit app integrating the entire system:
- **Sidebar:** Parameters (GA, TS, FLC bounds, SMT rules).
- **Tab 1: Dataset Exploration:** Details variables of the chosen `.mat` file.
- **Tab 2: Optimization Run:** Shows dynamic training charts.
- **Tab 3: Scheduling Gantt:** Displays the interactive Plotly timeline.
- **Tab 4: Solution Explainer:** Natural language explanations of bottlenecks, tardy jobs, and scheduling decisions.

---

## Verification Plan

### Automated Tests
Run unit tests checking:
1. Preprocessing (batch splitting quantities sum up to the original job quantity).
2. Decoding (start times at stage $w+1$ are $\ge$ end times at stage $w$ + transport time).
3. Fuzzy controller output bounds.

```bash
uv run pytest
```

### Manual Verification
1. Open Streamlit and run on `problem1.mat`. Verify the Plotly Gantt chart is drawn correctly and hovering over bars shows complete tooltips.
2. Confirm the KPIs display Makespan, Total Tardiness, and Setup Costs accurately.
3. Review the AI-style Solution Explainer text.
