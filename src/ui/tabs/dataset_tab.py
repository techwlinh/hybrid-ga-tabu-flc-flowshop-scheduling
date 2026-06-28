import os
import pickle
import pandas as pd
import streamlit as st
from typing import Any

from src.ui.sync import sync_workstations, sync_jobs


def render_dataset_tab(problem_instance: Any):
    """Renders Tab 1: Dataset Exploration."""
    st.subheader("Dataset Metadata & Parameters")

    jobs = problem_instance.jobs
    workstations = problem_instance.workstations
    setup_times = problem_instance.setup_times
    setup_costs = problem_instance.setup_costs

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Products (N)", len(jobs))
    with col2:
        st.metric("Total Workstations (M)", len(workstations))
    with col3:
        total_machines = sum(len(ws.machines) for ws in workstations)
        st.metric("Total SMT Lines (Machines)", total_machines)

    st.write("### Workstations Configuration")
    ws_info = []
    for ws in workstations:
        ws_info.append({"Workstation Name": ws.name, "Machine Count": len(ws.machines)})
    ws_df = pd.DataFrame(ws_info)

    edited_ws_df = st.data_editor(
        ws_df,
        num_rows="dynamic",
        column_config={
            "Workstation Name": st.column_config.TextColumn(required=True),
            "Machine Count": st.column_config.NumberColumn(
                min_value=1, max_value=10, step=1, required=True
            ),
        },
        use_container_width=True,
        key="ws_editor",
    )

    # Trigger synchronization for workstations if changed
    ws_needs_sync = False
    if len(edited_ws_df) != len(workstations):
        ws_needs_sync = True
    else:
        for idx, row in edited_ws_df.iterrows():
            if row["Workstation Name"] != workstations[idx].name or row[
                "Machine Count"
            ] != len(workstations[idx].machines):
                ws_needs_sync = True
                break

    if ws_needs_sync:
        sync_workstations(edited_ws_df, problem_instance)
        # Update local references
        jobs = problem_instance.jobs
        workstations = problem_instance.workstations
        setup_times = problem_instance.setup_times
        setup_costs = problem_instance.setup_costs

    # Transportation Times matrix
    st.write("### Transportation Times between Workstations (Matrix)")
    ws_names = [ws.name for ws in workstations]
    transport_df = pd.DataFrame(
        problem_instance.transport_matrix, index=ws_names, columns=ws_names
    )
    edited_transport_df = st.data_editor(
        transport_df, use_container_width=True, key="transport_editor"
    )
    problem_instance.transport_matrix = edited_transport_df.values

    st.write(
        "### Products Enriched Metadata (Scaled Quantities & Priorities & Eligibility)"
    )
    # Construct columns for all machines in the system
    machine_columns = []
    mach_tuples = []
    for ws in workstations:
        for mach in ws.machines:
            col_name = f"{ws.name} M{mach.id+1}"
            machine_columns.append(col_name)
            mach_tuples.append((ws.id, mach.id, col_name))

    jobs_data = []
    for job in jobs:
        row = {
            "Product ID": job.id,
            "Quantity (units)": job.quantity,
            "Priority Group": job.priority,
            "Material Arrival (min)": round(job.material_arrival_time, 2),
            "Due Date (min)": round(job.due_date, 2),
        }
        for ws in workstations:
            row[f"{ws.name} Unit PT (min)"] = round(job.unit_processing_times[ws.id], 2)
        for ws_id, mach_id, col_name in mach_tuples:
            is_eligible = mach_id in job.eligible_machines.get(ws_id, [])
            row[col_name] = is_eligible
        jobs_data.append(row)

    jobs_df = pd.DataFrame(jobs_data)

    # Build column configs
    col_config = {
        "Product ID": st.column_config.NumberColumn(disabled=True),
        "Quantity (units)": st.column_config.NumberColumn(min_value=1, required=True),
        "Priority Group": st.column_config.SelectboxColumn(
            options=[1, 2, 3, 4], required=True
        ),
        "Material Arrival (min)": st.column_config.NumberColumn(
            min_value=0.0, required=True
        ),
        "Due Date (min)": st.column_config.NumberColumn(min_value=0.0, required=True),
    }
    for ws in workstations:
        col_config[f"{ws.name} Unit PT (min)"] = st.column_config.NumberColumn(min_value=0.0, required=True)
    for col_name in machine_columns:
        col_config[col_name] = st.column_config.CheckboxColumn(required=True)

    edited_jobs_df = st.data_editor(
        jobs_df,
        column_config=col_config,
        use_container_width=True,
        key="jobs_editor",
    )

    # Trigger synchronization for jobs
    sync_jobs(edited_jobs_df, problem_instance, mach_tuples)

    # Setup matrix configuration
    st.write("### Sequence-Dependent Changeover Setup Matrix")
    ws_select = st.selectbox(
        "Select Workstation Stage for Setup Matrix",
        options=workstations,
        format_func=lambda w: w.name,
    )

    if ws_select is not None:
        w_id = ws_select.id
        product_ids = [f"Product {j.id}" for j in jobs]

        col_setup1, col_setup2 = st.columns(2)
        with col_setup1:
            st.write(f"**Setup Times (minutes) at {ws_select.name}**")
            setup_times_df = pd.DataFrame(
                setup_times[:, :, w_id], index=product_ids, columns=product_ids
            )
            edited_setup_times = st.data_editor(
                setup_times_df, use_container_width=True, key=f"setup_times_{w_id}"
            )
            setup_times[:, :, w_id] = edited_setup_times.values

        with col_setup2:
            st.write(f"**Setup Costs ($) at {ws_select.name}**")
            setup_costs_df = pd.DataFrame(
                setup_costs[:, :, w_id], index=product_ids, columns=product_ids
            )
            edited_setup_costs = st.data_editor(
                setup_costs_df, use_container_width=True, key=f"setup_costs_{w_id}"
            )
            setup_costs[:, :, w_id] = edited_setup_costs.values


    # Save version section
    st.write("---")
    st.write("### 💾 Save Current Dataset as a Version")
    with st.form("save_version_form"):
        col_vname, col_vbtn = st.columns([3, 1])
        with col_vname:
            version_name = st.text_input(
                "Enter Version Name",
                value="",
                placeholder="e.g. s1_v1_modified",
                help="Filename under which to save this customized problem instance.",
            ).strip()
        with col_vbtn:
            st.write("")  # Spacer
            st.write("")  # Spacer
            submit_save = st.form_submit_button("Save Dataset Version")

        if submit_save:
            if not version_name:
                st.error("Please enter a valid version name.")
            else:
                safe_name = "".join(
                    c for c in version_name if c.isalnum() or c in ("-", "_")
                ).strip()
                if not safe_name:
                    st.error(
                        "Invalid version name. Use alphanumeric characters, dashes, and underscores."
                    )
                else:
                    filename = f"{safe_name}.pkl"
                    filepath = os.path.join("data_versions", filename)
                    try:
                        with open(filepath, "wb") as f:
                            pickle.dump(problem_instance, f)
                        st.success(
                            f"Successfully saved version as `{filename}` in `data_versions/`!"
                        )
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error saving version: {e}")
