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
        workstations: List[Workstation]
    ) -> go.Figure:
        """
        Creates an interactive Plotly Gantt chart.
        - Rows: Grouped by Workstation and Machine ID.
        - Color: Distinct colors for each Job, and gray pattern/color for Setup activities.
        - Hover: Displays job IDs, batch details, quantities, priority, wait times, and setup cost/time.
        """
        df_entries = []
        num_workstations = len(workstations)
        
        # Color mapping for jobs
        # Create a set of distinct colors for jobs and add gray for setups
        unique_jobs = list(jobs_dict.keys())
        # Generate color scale
        colors = px.colors.qualitative.Alphabet
        job_colors = {job_id: colors[i % len(colors)] for i, job_id in enumerate(unique_jobs)}
        
        for entry in result.entries:
            job = jobs_dict[entry.job_id]
            machine_label = f"WS {entry.workstation_id+1} - Machine {entry.machine_id+1}"
            
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
                    "Color": "#94A3B8"  # Slate/Gray for setups
                })
                
            # 2. Add Processing Entry
            # Calculate batch tardiness if it is the final workstation
            tardiness = 0.0
            if entry.workstation_id == num_workstations - 1:
                tardiness = max(0.0, entry.end_time - job.due_date)
                
            df_entries.append({
                "Machine": machine_label,
                "Start": SMTGanttChart._time_to_dt(entry.start_time),
                "Finish": SMTGanttChart._time_to_dt(entry.end_time),
                "Activity": f"Job {entry.job_id}",
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

        # Sort entries by Workstation and Machine label for clean Gantt rows
        df_entries.sort(key=lambda x: x["Machine"])
        
        # Build custom timeline chart
        fig = px.timeline(
            df_entries,
            x_start="Start",
            x_end="Finish",
            y="Machine",
            color="Activity",
            color_discrete_map={
                "Machine Setup/Changeover": "#94A3B8",
                **{f"Job {jid}": job_colors[jid] for jid in unique_jobs}
            },
            title="Interactive HFS Gantt Chart (Parallel SMT Lines)"
        )
        
        # Configure layout and custom tooltip template
        fig.update_yaxes(categoryorder="array", categoryarray=sorted(list(set(x["Machine"] for x in df_entries))))
        
        hover_template = (
            "<b>Activity:</b> %{customdata[0]}<br>"
            "<b>Batch ID:</b> %{customdata[1]}<br>"
            "<b>Job ID:</b> %{customdata[2]}<br>"
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
            # Match elements in trace
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

        fig.update_layout(
            xaxis_title="Timeline (HH:MM:SS from Reference Day 1)",
            yaxis_title="SMT Workstation & Machine Lines",
            hoverlabel=dict(bgcolor="white", font_size=11, font_family="JetBrains Mono"),
            legend_title_text="Job Batches & Changeovers",
            margin=dict(l=150, r=20, t=50, b=50),
            height=600
        )
        
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
