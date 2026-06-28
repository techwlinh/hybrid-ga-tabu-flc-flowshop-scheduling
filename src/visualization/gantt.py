import datetime
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Dict, Any
from src.models import ScheduleResult, Job, Workstation

class SMTGanttChart:
    @staticmethod
    def _time_to_dt(time_units: float) -> datetime.datetime:
        """Converts numerical time units (minutes) to datetime objects for Plotly timeline."""
        base_dt = datetime.datetime(2026, 1, 1, 0, 0, 0)
        return base_dt + datetime.timedelta(minutes=time_units)

    @staticmethod
    def plot_interactive_gantt(
        result: ScheduleResult,
        jobs_dict: Dict[int, Job],
        workstations: List[Workstation],
        transport_matrix: np.ndarray = None,
        max_time_scale: float = None
    ) -> go.Figure:
        """
        Creates an interactive Plotly Gantt chart.
        - Rows: Grouped by Workstation and Machine ID using hierarchical multi-category y-axis.
        - Color: Distinct colors for each Job, gray for Setup activities, and light blue for Transportation transit.
        - Hover: Displays job IDs, batch details, quantities, priority, wait times, and setup cost/time.
        """
        df_entries = []
        num_workstations = len(workstations)
        
        # Color mapping for jobs
        unique_jobs = list(jobs_dict.keys())
        colors = px.colors.qualitative.Alphabet
        job_colors = {job_id: colors[i % len(colors)] for i, job_id in enumerate(unique_jobs)}
        
        # Pre-add transparent dummy entries for all machines to ensure they are visible even if idle
        for ws in workstations:
            for mach in ws.machines:
                machine_label = f"{ws.name} - Machine {mach.id+1}"
                df_entries.append({
                    "Machine": machine_label,
                    "Start": SMTGanttChart._time_to_dt(0.0),
                    "Finish": SMTGanttChart._time_to_dt(0.0),
                    "Activity": "Idle/Non-allocated",
                    "Job ID": "N/A",
                    "Batch ID": "N/A",
                    "Type": "Idle",
                    "Quantity": 0,
                    "Priority": "N/A",
                    "Due Date": "N/A",
                    "Tardiness": "N/A",
                    "Setup Time": "N/A",
                    "Setup Cost": "N/A",
                    "Waiting Time": "N/A",
                    "Color": "rgba(0,0,0,0)"
                })
                
        # Group entries by batch to track completion times of previous workstations for transit calculation
        batch_runs = {}
        for entry in result.entries:
            batch_runs.setdefault(entry.batch_id, []).append(entry)
        for b_id in batch_runs:
            batch_runs[b_id].sort(key=lambda x: x.workstation_id)
            
        for entry in result.entries:
            job = jobs_dict[entry.job_id]
            ws = workstations[entry.workstation_id]
            mach = ws.machines[entry.machine_id]
            machine_label = f"{ws.name} - Machine {mach.id+1}"
            
            # 0. Add Transportation Entry (if stage w > 0)
            if entry.workstation_id > 0:
                runs = batch_runs.get(entry.batch_id, [])
                prev_run = next((r for r in runs if r.workstation_id == entry.workstation_id - 1), None)
                if prev_run is not None:
                    prev_stage_end = prev_run.end_time
                    t_time = transport_matrix[entry.workstation_id - 1, entry.workstation_id] if transport_matrix is not None else 10.0
                    if t_time > 0:
                        df_entries.append({
                            "Machine": machine_label,
                            "Start": SMTGanttChart._time_to_dt(prev_stage_end),
                            "Finish": SMTGanttChart._time_to_dt(prev_stage_end + t_time),
                            "Activity": f"Product {entry.job_id} Transit",
                            "Job ID": str(entry.job_id),
                            "Batch ID": entry.batch_id,
                            "Type": "Transportation",
                            "Quantity": int(job.quantity),
                            "Priority": str(job.priority),
                            "Due Date": f"{job.due_date:.1f} min",
                            "Tardiness": "N/A",
                            "Setup Time": "N/A",
                            "Setup Cost": "N/A",
                            "Waiting Time": "N/A",
                            "Color": "#38BDF8"  # Light blue for transport transit
                        })
            
            # 1. Add Setup Entry (if setup_time > 0)
            if entry.setup_time > 0:
                setup_start = entry.start_time - entry.setup_time
                df_entries.append({
                    "Machine": machine_label,
                    "Start": SMTGanttChart._time_to_dt(setup_start),
                    "Finish": SMTGanttChart._time_to_dt(entry.start_time),
                    "Activity": "Machine Setup/Changeover",
                    "Job ID": "N/A",
                    "Batch ID": entry.batch_id,
                    "Type": "Setup",
                    "Quantity": 0,
                    "Priority": "N/A",
                    "Due Date": "N/A",
                    "Tardiness": 0.0,
                    "Setup Time": f"{entry.setup_time:.1f} min",
                    "Setup Cost": f"${entry.setup_cost:.1f}",
                    "Waiting Time": f"{entry.waiting_time:.1f} min",
                    "Color": "#94A3B8"
                })
                
            # 2. Add Processing Entry
            tardiness = 0.0
            if entry.workstation_id == num_workstations - 1:
                tardiness = max(0.0, entry.end_time - job.due_date)
                
            df_entries.append({
                "Machine": machine_label,
                "Start": SMTGanttChart._time_to_dt(entry.start_time),
                "Finish": SMTGanttChart._time_to_dt(entry.end_time),
                "Activity": f"Product {entry.job_id}",
                "Job ID": str(entry.job_id),
                "Batch ID": entry.batch_id,
                "Type": "Processing",
                "Quantity": int(job.quantity),
                "Priority": str(job.priority),
                "Due Date": f"{job.due_date:.1f} min",
                "Tardiness": f"{tardiness:.1f} min",
                "Setup Time": "N/A",
                "Setup Cost": "N/A",
                "Waiting Time": f"{entry.waiting_time:.1f} min",
                "Color": job_colors[entry.job_id]
            })

        if not df_entries:
            return go.Figure()
            
        # Build colors discrete map
        color_discrete_map = {
            "Idle/Non-allocated": "rgba(0,0,0,0)",
            "Machine Setup/Changeover": "#94A3B8"
        }
        for jid in unique_jobs:
            color_discrete_map[f"Product {jid}"] = job_colors[jid]
            color_discrete_map[f"Product {jid} Transit"] = "#38BDF8"
            
        # Build timeline chart
        fig = px.timeline(
            df_entries,
            x_start="Start",
            x_end="Finish",
            y="Machine",
            color="Activity",
            color_discrete_map=color_discrete_map,
            title="Interactive HFS Gantt Chart (Parallel SMT Lines)"
        )
        
        # Configure layout and custom tooltip template
        hover_template = (
            "<b>Activity:</b> %{customdata[0]}<br>"
            "<b>Batch ID:</b> %{customdata[1]}<br>"
            "<b>Product ID:</b> %{customdata[2]}<br>"
            "<b>Quantity:</b> %{customdata[3]} units<br>"
            "<b>Priority:</b> Group %{customdata[4]}<br>"
            "<b>Due Date:</b> %{customdata[5]}<br>"
            "<b>Tardiness:</b> %{customdata[6]}<br>"
            "<b>Setup Time:</b> %{customdata[7]}<br>"
            "<b>Setup Cost:</b> %{customdata[8]}<br>"
            "<b>Idle/Wait Time:</b> %{customdata[9]}<br>"
            "<b>Start:</b> %{x|%H:%M:%S}<br>"
            "<extra></extra>"
        )
        
        # Package custom data fields for tooltips
        for trace in fig.data:
            trace_activity = trace.name
            trace_entries = [e for e in df_entries if e["Activity"] == trace_activity]
            
            custom_data = []
            for e in trace_entries:
                custom_data.append([
                    e["Activity"],
                    e["Batch ID"],
                    e["Job ID"],
                    e["Quantity"],
                    e["Priority"],
                    e["Due Date"],
                    e["Tardiness"],
                    e["Setup Time"],
                    e["Setup Cost"],
                    e["Waiting Time"]
                ])
            trace.customdata = custom_data
            trace.hovertemplate = hover_template

        # Multi-category y-axis grouping mapping
        machine_to_multicategory = {}
        for ws in workstations:
            for mach in ws.machines:
                key = f"{ws.name} - Machine {mach.id+1}"
                machine_to_multicategory[key] = (ws.name, f"M{mach.id+1}")
                
        # Group and configure category order
        all_workstations_list = []
        all_machines_list = []
        for ws in reversed(workstations):
            for mach in reversed(ws.machines):
                all_workstations_list.append(ws.name)
                all_machines_list.append(f"M{mach.id+1}")
                
        # Convert trace.y to hierarchical list of lists
        for trace in fig.data:
            ws_list = []
            mach_list = []
            for y_val in trace.y:
                ws_name, mach_name = machine_to_multicategory.get(y_val, ("N/A", "N/A"))
                ws_list.append(ws_name)
                mach_list.append(mach_name)
            trace.y = [ws_list, mach_list]
            
        fig.update_yaxes(
            categoryorder="array",
            categoryarray=[all_workstations_list, all_machines_list]
        )

        fig.update_layout(
            xaxis_title="Timeline (HH:MM:SS from Reference Day 1)",
            yaxis_title="SMT Workstations & Machines",
            hoverlabel=dict(bgcolor="white", font_size=11, font_family="JetBrains Mono"),
            legend_title_text="Job Batches & Changeovers",
            margin=dict(l=220, r=20, t=50, b=50),
            height=600
        )
        
        # Add range slider and adjust initial view limits if specified
        fig.update_xaxes(rangeslider_visible=True)
        if max_time_scale is not None:
            fig.update_xaxes(range=[SMTGanttChart._time_to_dt(0.0), SMTGanttChart._time_to_dt(max_time_scale)])
            
        return fig

    @staticmethod
    def plot_static_gantt(
        result: ScheduleResult,
        jobs_dict: Dict[int, Job],
        workstations: List[Workstation],
        output_path: str = None
    ):
        """
        Generates a static Matplotlib Gantt chart and optionally saves it.
        Useful for command-line runs and automated reporting.
        """
        fig, ax = plt.subplots(figsize=(15, 8))
        
        # Unique machines list
        machines_list = []
        for ws in workstations:
            for mach in ws.machines:
                machines_list.append((ws.id, mach.id, f"WS {ws.id+1} - M{mach.id+1}"))
                
        # Sort machine strings
        machines_list.sort(key=lambda x: (x[0], x[1]))
        mach_to_y = {f"WS {w_id+1} - M{m_id+1}": idx for idx, (w_id, m_id, _) in enumerate(machines_list)}
        
        # Distinct colors mapping
        from matplotlib import colormaps
        unique_jobs = list(jobs_dict.keys())
        colormap = colormaps["tab20"]
        job_colors = {job_id: colormap((i % 20) / 20.0) for i, job_id in enumerate(unique_jobs)}
        
        # Plot bars
        for entry in result.entries:
            y_lbl = f"WS {entry.workstation_id+1} - M{entry.machine_id+1}"
            y_pos = mach_to_y[y_lbl]
            
            # Setup bar (if setup_time > 0)
            if entry.setup_time > 0:
                setup_start = entry.start_time - entry.setup_time
                ax.barh(y_pos, entry.setup_time, left=setup_start, color="gray", alpha=0.5, hatch="//", edgecolor="black")
                
            # Processing bar
            color = job_colors[entry.job_id]
            ax.barh(y_pos, entry.end_time - entry.start_time, left=entry.start_time, color=color, edgecolor="black", alpha=0.8)
            
            # Print batch text label inside the bar
            center = (entry.start_time + entry.end_time) / 2
            duration = entry.end_time - entry.start_time
            if duration > 10.0:  # Only label if bar is wide enough
                ax.text(center, y_pos, entry.batch_id, ha="center", va="center", color="white", fontsize=8, weight="bold")
                
        # Setup axes
        ax.set_yticks(range(len(machines_list)))
        ax.set_yticklabels([label for _, _, label in machines_list])
        ax.set_xlabel("Time (minutes)")
        ax.set_ylabel("Workstation Machine Lines")
        ax.set_title("SMT Flow Shop Schedule Gantt Chart")
        ax.grid(axis="x", linestyle="--", alpha=0.5)
        
        # Custom Legend
        legend_patches = []
        for job_id in unique_jobs:
            patch = mpatches.Patch(color=job_colors[job_id], label=f"Job {job_id} (Q={jobs_dict[job_id].quantity})")
            legend_patches.append(patch)
        # Add setup patch
        setup_patch = mpatches.Patch(color="gray", alpha=0.5, hatch="//", label="Changeover Setup")
        legend_patches.append(setup_patch)
        
        ax.legend(handles=legend_patches, bbox_to_anchor=(1.02, 1), loc="upper left", title="Legend")
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300)
            plt.close()
        else:
            plt.show()
