import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple
from src.models import ScheduleResult, Job, Workstation

def get_scheduling_stats_and_charts(
    result: ScheduleResult,
    jobs: List[Job],
    workstations: List[Workstation]
) -> Dict[str, Any]:
    """
    Computes key performance metrics (overall and per workstation LBE, Smoothness Index,
    job tardiness rates) and prepares dataframes for visualization.
    """
    # 1. Base statistics
    total_jobs = len(jobs)
    
    if not result or not result.entries:
        return {
            "lbe_overall": 0.0,
            "si_overall": 0.0,
            "tardy_rate": 0.0,
            "total_jobs": total_jobs,
            "tardy_jobs_count": 0,
            "df_machines": pd.DataFrame(),
            "df_jobs": pd.DataFrame(),
            "lbe_per_workstation": {},
            "si_per_workstation": {},
            "ws_utilization": {}
        }
        
    makespan = result.makespan
    
    # 2. Line Balancing Metrics (LBE & SI)
    # Get machine utilization dictionary
    machine_util = result.machine_utilization
    utils = list(machine_util.values())
    
    max_util = max(utils) if utils else 0.0
    mean_util = np.mean(utils) if utils else 0.0
    
    if max_util > 0.0:
        lbe_overall = (mean_util / max_util) * 100.0
    else:
        lbe_overall = 0.0
        
    if utils:
        si_overall = np.sqrt(np.mean([(max_util - u)**2 for u in utils])) * 100.0
    else:
        si_overall = 0.0
        
    lbe_per_workstation = {}
    si_per_workstation = {}
    
    for ws in workstations:
        ws_utils = []
        for mach in ws.machines:
            key = f"W{ws.id+1}_M{mach.id+1}"
            if key in machine_util:
                ws_utils.append(machine_util[key])
        
        ws_max = max(ws_utils) if ws_utils else 0.0
        ws_mean = np.mean(ws_utils) if ws_utils else 0.0
        
        if ws_max > 0.0:
            lbe_per_workstation[ws.id] = (ws_mean / ws_max) * 100.0
            si_per_workstation[ws.id] = np.sqrt(np.mean([(ws_max - u)**2 for u in ws_utils])) * 100.0
        else:
            lbe_per_workstation[ws.id] = 0.0
            si_per_workstation[ws.id] = 0.0

    # 3. Machine Workload Breakdown
    machine_data = []
    for ws in workstations:
        for mach in ws.machines:
            m_label = f"{ws.name} - Machine {mach.id+1}"
            m_entries = [e for e in result.entries if e.workstation_id == ws.id and e.machine_id == mach.id]
            
            setup_time = sum(e.setup_time for e in m_entries)
            processing_time = sum((e.end_time - e.start_time) for e in m_entries)
            
            # Idle time is makespan minus busy time
            idle_time = max(0.0, makespan - (setup_time + processing_time))
            
            machine_data.append({
                "Machine": m_label,
                "Workstation": ws.name,
                "Workstation ID": ws.id,
                "Time (min)": processing_time,
                "Activity": "Processing"
            })
            machine_data.append({
                "Machine": m_label,
                "Workstation": ws.name,
                "Workstation ID": ws.id,
                "Time (min)": setup_time,
                "Activity": "Setup Changeover"
            })
            machine_data.append({
                "Machine": m_label,
                "Workstation": ws.name,
                "Workstation ID": ws.id,
                "Time (min)": idle_time,
                "Activity": "Idle / Non-allocated"
            })
            
    df_machines = pd.DataFrame(machine_data)
    
    # 4. Job Tardiness & Completion
    # Find completion time of each job (at final stage M-1)
    num_ws = len(workstations)
    job_completions = {}
    for entry in result.entries:
        if entry.workstation_id == num_ws - 1:
            job_completions[entry.job_id] = max(
                job_completions.get(entry.job_id, 0.0), entry.end_time
            )
            
    job_data = []
    tardy_jobs_count = 0
    
    for job in jobs:
        comp_time = job_completions.get(job.id, 0.0)
        tardiness = max(0.0, comp_time - job.due_date)
        is_tardy = tardiness > 0.0
        if is_tardy:
            tardy_jobs_count += 1
            
        job_data.append({
            "Job ID": f"Job {job.id}",
            "Job Number": job.id,
            "Priority": f"Priority {job.priority}",
            "Priority Int": job.priority,
            "Due Date (min)": job.due_date,
            "Completion Time (min)": comp_time,
            "Tardiness (min)": tardiness,
            "Status": "Tardy" if is_tardy else "On-Time"
        })
        
    df_jobs = pd.DataFrame(job_data)
    tardy_rate = (tardy_jobs_count / total_jobs) * 100.0 if total_jobs > 0 else 0.0
    
    # 5. Workstation utilization dictionary
    ws_utilization = {ws.id: result.workstation_utilization.get(ws.id, 0.0) * 100.0 for ws in workstations}
    
    return {
        "lbe_overall": lbe_overall,
        "si_overall": si_overall,
        "tardy_rate": tardy_rate,
        "total_jobs": total_jobs,
        "tardy_jobs_count": tardy_jobs_count,
        "df_machines": df_machines,
        "df_jobs": df_jobs,
        "lbe_per_workstation": lbe_per_workstation,
        "si_per_workstation": si_per_workstation,
        "ws_utilization": ws_utilization
    }


def plot_machine_workload(df_machines: pd.DataFrame, workstation_filter: str = "All") -> go.Figure:
    """Creates an interactive horizontal stacked bar chart showing Processing, Setup, and Idle times."""
    if df_machines.empty:
        return go.Figure()
        
    df_plot = df_machines.copy()
    if workstation_filter != "All":
        df_plot = df_plot[df_plot["Workstation"] == workstation_filter]
        
    color_map = {
        "Processing": "#10B981",          # Emerald
        "Setup Changeover": "#F59E0B",    # Amber
        "Idle / Non-allocated": "#9CA3AF"  # Gray
    }
    
    # Sort machines in logical order
    # Grouped by workstation, then by machine
    unique_machines = list(df_plot["Machine"].unique())
    # Reverse order so they appear from top to bottom
    category_order = unique_machines[::-1]
    
    fig = px.bar(
        df_plot,
        y="Machine",
        x="Time (min)",
        color="Activity",
        color_discrete_map=color_map,
        orientation="h",
        category_orders={"Machine": category_order},
        title="Machine Workload Breakdown (Processing, Setup, Idle)",
        labels={"Time (min)": "Duration (minutes)", "Machine": "SMT Machine"}
    )
    
    fig.update_layout(
        barmode="stack",
        margin=dict(l=20, r=20, t=50, b=40),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        height=max(350, len(unique_machines) * 35 + 100)
    )
    fig.update_yaxes(ticksuffix=" ")
    return fig


def plot_job_performance_scatter(df_jobs: pd.DataFrame) -> go.Figure:
    """Creates a scatter plot comparing job completion times with due dates."""
    if df_jobs.empty:
        return go.Figure()
        
    color_map = {
        "On-Time": "#10B981",  # Emerald
        "Tardy": "#EF4444"     # Red
    }
    
    fig = px.scatter(
        df_jobs,
        x="Due Date (min)",
        y="Completion Time (min)",
        color="Status",
        color_discrete_map=color_map,
        symbol="Priority",
        hover_name="Job ID",
        hover_data={
            "Due Date (min)": ":.1f",
            "Completion Time (min)": ":.1f",
            "Tardiness (min)": ":.1f",
            "Priority": True,
            "Status": False
        },
        title="Job Completion Time vs. Due Date Deadline",
        labels={
            "Due Date (min)": "Due Date Deadline (min)",
            "Completion Time (min)": "Actual Completion Time (min)"
        }
    )
    
    # Calculate a suitable range for the reference line
    max_val = max(df_jobs["Due Date (min)"].max(), df_jobs["Completion Time (min)"].max()) * 1.1 if not df_jobs.empty else 100
    
    # Add diagonal reference line y = x
    fig.add_shape(
        type="line",
        x0=0, y0=0, x1=max_val, y1=max_val,
        line=dict(color="#6B7280", width=2, dash="dash"),
        name="Due Date Deadline"
    )
    
    fig.update_layout(
        margin=dict(l=20, r=20, t=50, b=40),
        xaxis=dict(range=[0, max_val]),
        yaxis=dict(range=[0, max_val]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        height=450
    )
    return fig


def plot_job_status_donut(df_jobs: pd.DataFrame) -> go.Figure:
    """Creates a donut chart representing the proportion of on-time vs tardy jobs."""
    if df_jobs.empty:
        return go.Figure()
        
    status_counts = df_jobs["Status"].value_counts().reset_index()
    status_counts.columns = ["Status", "Count"]
    
    color_map = {
        "On-Time": "#10B981",  # Emerald
        "Tardy": "#EF4444"     # Red
    }
    
    fig = px.pie(
        status_counts,
        names="Status",
        values="Count",
        color="Status",
        color_discrete_map=color_map,
        hole=0.4,
        title="Delivery Schedule Compliance Rate"
    )
    
    fig.update_layout(
        margin=dict(l=20, r=20, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5),
        height=400
    )
    fig.update_traces(textposition='inside', textinfo='percent+label')
    return fig


def plot_workstation_utilization(ws_utilization: Dict[int, float], workstations: List[Workstation]) -> go.Figure:
    """Creates a bar chart showing the average machine utilization rate for each workstation."""
    if not ws_utilization:
        return go.Figure()
        
    data = []
    for ws in workstations:
        data.append({
            "Workstation": ws.name,
            "Utilization (%)": ws_utilization.get(ws.id, 0.0)
        })
    df_ws = pd.DataFrame(data)
    
    # Calculate a color gradient or highlight the bottleneck (highest utilization)
    max_idx = df_ws["Utilization (%)"].idxmax() if not df_ws.empty else -1
    colors = ["#3B82F6"] * len(df_ws) # Default Blue
    if max_idx >= 0:
        colors[max_idx] = "#EF4444" # Highlight bottleneck in red
        
    df_ws["Color"] = colors
    
    fig = px.bar(
        df_ws,
        x="Workstation",
        y="Utilization (%)",
        text=df_ws["Utilization (%)"].apply(lambda x: f"{x:.1f}%"),
        title="Workstation Average Machine Utilization (%)",
        labels={"Utilization (%)": "Average Utilization (%)", "Workstation": "Workstation Stage"}
    )
    
    fig.update_traces(
        marker_color=colors,
        textposition="outside"
    )
    
    fig.update_layout(
        margin=dict(l=20, r=20, t=50, b=40),
        yaxis=dict(range=[0, 115]), # Extra space for label text
        height=400
    )
    return fig
