import datetime
import os
import shutil
import tempfile
import traceback

import pandas as pd
from pyswmm import Nodes, Simulation
from swmm_api.input_file import SwmmInput, section_labels as sections


def get_user_input(prompt, default_value=None):
    default_text = f" (default: {default_value})" if default_value not in (None, '') else ""
    user_input = input(f"{prompt}{default_text}: ").strip()
    return user_input if user_input else default_value


def normalize_user_path(path_value):
    if path_value in (None, ''):
        return None
    path_value = str(path_value).strip().strip('"').strip("'")
    return os.path.abspath(os.path.expandvars(os.path.expanduser(path_value)))


def format_elapsed_sim_time(start_time, current_time):
    day_offset = (current_time.date() - start_time.date()).days
    time_label = current_time.strftime('%H:%M:%S')
    if day_offset <= 0:
        return time_label
    return f'+{day_offset}d {time_label}'


def prompt_for_existing_swmm_model(current_path=None):
    current_path = normalize_user_path(current_path)
    if current_path and os.path.isfile(current_path):
        return current_path

    if current_path:
        print(f"Configured SWMM model path does not exist: {current_path}")
    print("A private SWMM .inp model is required for SWMM simulation options.")

    while True:
        candidate = normalize_user_path(get_user_input("Enter your SWMM model .inp file path"))
        if not candidate:
            print("No SWMM model path was provided.")
            continue
        if not candidate.lower().endswith('.inp'):
            print(f"Expected a SWMM .inp file, got: {candidate}")
            continue
        if not os.path.isfile(candidate):
            print(f"File does not exist: {candidate}")
            continue
        return candidate


def update_simulation_parameters(inp, raingage_name, rainfall_event, imperviousness_para2, minimum_rate, config, output_path=None):
    inp[sections.RAINGAGES][raingage_name]['timeseries'] = rainfall_event
    form = 'CUMULATIVE' if rainfall_event in config['RAINFALL_EVENTS_SYNTHETIC'] else 'VOLUME'
    interval = '0:06' if rainfall_event in config['RAINFALL_EVENTS_SYNTHETIC'] else '0:05'
    inp[sections.RAINGAGES][raingage_name]['interval'] = interval
    inp[sections.RAINGAGES][raingage_name]['form'] = form

    for subcatchment_name in inp[sections.SUBCATCHMENTS]:
        inp[sections.SUBCATCHMENTS][subcatchment_name]['imperviousness'] = imperviousness_para2

    for infiltration_name in inp[sections.INFILTRATION]:
        inp[sections.INFILTRATION][infiltration_name]['rate_min'] = minimum_rate

    inp[sections.OPTIONS]['REPORT_STEP'] = '00:05:00'
    output_path = output_path or config['SIMULATION_PATH']
    inp.write_file(output_path)
    return output_path


def get_model_info(simulation_path):
    with Simulation(simulation_path) as sim:
        sim.step_advance(300)
        system_units = sim.system_units
        flow_units = sim.flow_units
        engine_version = sim.engine_version
    inp = SwmmInput.read_file(simulation_path)
    report_step = inp[sections.OPTIONS]['REPORT_STEP']
    wet_runoff_step = inp[sections.OPTIONS]['WET_STEP']
    dry_runoff_step = inp[sections.OPTIONS]['DRY_STEP']
    routing_step = inp[sections.OPTIONS]['ROUTING_STEP']
    print(
        f"System units: {system_units}\n"
        f"Flow units: {flow_units}\n"
        f"Engine version: {engine_version}\n"
        f"Report_step: {report_step}\n"
        f"Wet_runoff_step: {wet_runoff_step}\n"
        f"Dry_runoff_step: {dry_runoff_step}\n"
        f"Routing_step: {routing_step}"
    )


def get_peak_flow_at_node(simulation_path, node_id):
    with Simulation(simulation_path) as sim:
        sim.step_advance(300)
        nodes = Nodes(sim)
        peak_flow = 0
        peak_time = None
        for _ in sim:
            total_inflow = nodes[node_id].total_inflow
            if total_inflow > peak_flow:
                peak_flow = total_inflow
                peak_time = sim.current_time
    return peak_flow, peak_time


def get_flows_at_time(simulation_path, target_time):
    flows = {}
    with Simulation(simulation_path) as sim:
        sim.step_advance(300)
        nodes = Nodes(sim)
        for _ in sim:
            if sim.current_time == target_time:
                for node in nodes:
                    flows[node.nodeid] = node.total_inflow
                break
    return flows


def combine_csv_files(csv_directory, current_dir):
    if not os.path.exists(csv_directory):
        print(f"Directory '{csv_directory}' does not exist. Creating...")
        os.makedirs(csv_directory)
        print(f"Directory created: {csv_directory}")
        return None

    csv_files = [os.path.join(csv_directory, file) for file in os.listdir(csv_directory) if file.endswith('.csv')]
    csv_files.sort()
    if not csv_files:
        print("No CSV files found.")
        return

    first_df = pd.read_csv(csv_files[0])
    node_id_column = first_df.columns[0]
    combined_df = first_df.copy()

    for csv_file in csv_files[1:]:
        df = pd.read_csv(csv_file)
        df = df.drop(columns=[node_id_column], errors='ignore')
        combined_df = pd.concat([combined_df, df], axis=1)

    output_csv_path = os.path.join(current_dir, '..', 'SWMM_Model', 'combined_results.csv')
    output_combined_csv_path = get_user_input(
        "Enter the output path (directory and filename) for the merged file, press Enter to skip",
        output_csv_path
    )

    output_dir = os.path.dirname(output_combined_csv_path)
    if not os.path.exists(output_dir):
        print(f"Output Directory '{output_dir}' does not exist. Creating...")
        os.makedirs(output_dir)
        print(f"Directory created: {output_dir}")
    combined_df.to_csv(output_combined_csv_path, index=False)
    return output_combined_csv_path


def get_flows_at_target_time(simulation_path, rainfall_data, imperviousness_data, minimum_rate_data, config):
    t = get_user_input("Enter target time point (format HH:MM:SS), press Enter to skip", '12:00:00')
    target_time = datetime.datetime.strptime(t, '%H:%M:%S').time()
    inp = SwmmInput.read_file(simulation_path)
    for raingage_name in inp[sections.RAINGAGES]:
        for rainfall in rainfall_data:
            for imperviousness_para2 in imperviousness_data:
                for minimum_rate in minimum_rate_data:
                    temp_dir = tempfile.mkdtemp()
                    temp_inp_path = os.path.join(temp_dir, 'temp_model.inp')
                    try:
                        inp_copy = inp.copy()
                        update_simulation_parameters(
                            inp_copy, raingage_name, rainfall, imperviousness_para2, minimum_rate, config, temp_inp_path
                        )
                        flows = {}
                        with Simulation(temp_inp_path) as sim:
                            sim.step_advance(300)
                            nodes = Nodes(sim)
                            for _ in sim:
                                if sim.current_time.time() == target_time:
                                    for node in nodes:
                                        flows[node.nodeid] = node.total_inflow
                                    break
                    finally:
                        shutil.rmtree(temp_dir, ignore_errors=True)
                    print(f"Flow Rate at {target_time}: {flows}")

                    df_node = pd.DataFrame(list(flows.items()), columns=['Node_ID', f'Flow Rate at {target_time}'])
                    output_csv_path = (
                        config['OUTPUT_CSV_BASE_PATH']
                        + f'{rainfall}_impervious_{imperviousness_para2}_rate_{minimum_rate}_target_time.csv'
                    )
                    output_dir = os.path.dirname(output_csv_path)
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)

                    df_node.to_csv(output_csv_path, index=False)
                    print(
                        f"Results for {rainfall} (Imperviousness: {imperviousness_para2}, Minimum Rate: {minimum_rate}) "
                        f"have been saved to {output_csv_path}"
                    )
    if input("Merge all CSV files? (y/n, default: n): ").strip().lower() == 'y':
        csv_directory = get_user_input(
            "Enter the folder path containing all CSV files to merge, press Enter to skip",
            config['CSV_DIRECTORY']
        )
        output_combined_csv_path = combine_csv_files(csv_directory, config['CURRENT_DIR'])
        print("Merge complete. File saved to", output_combined_csv_path)
    else:
        print("Merge skipped.")


def get_flows_on_target_nodes_at_peak_time(simulation_path, target_node_ids, rainfall_data, imperviousness_data, minimum_rate_data, config):
    inp = SwmmInput.read_file(simulation_path)
    for raingage_name in inp[sections.RAINGAGES]:
        for rainfall in rainfall_data:
            for imperviousness_para2 in imperviousness_data:
                for minimum_rate in minimum_rate_data:
                    temp_dir = tempfile.mkdtemp()
                    temp_inp_path = os.path.join(temp_dir, 'temp_model.inp')
                    try:
                        inp_copy = inp.copy()
                        update_simulation_parameters(
                            inp_copy, raingage_name, rainfall, imperviousness_para2, minimum_rate, config, temp_inp_path
                        )
                        peak_flows = {}
                        peak_times = {}

                        for node_id in target_node_ids:
                            peak_flow, peak_time = get_peak_flow_at_node(temp_inp_path, node_id)
                            peak_flows[node_id] = peak_flow
                            peak_times[node_id] = peak_time

                        for node_id in target_node_ids:
                            print(f"Node {node_id} peak flow: {peak_flows[node_id]} at {peak_times[node_id].strftime('%H:%M:%S')}")

                        all_other_nodes_flow = {}
                        for node_id, peak_time in peak_times.items():
                            other_nodes_flow = get_flows_at_time(temp_inp_path, peak_time)
                            all_other_nodes_flow[node_id] = other_nodes_flow
                            print(f"Flow rates at {rainfall} {peak_times[node_id].strftime('%H:%M:%S')} for Node {node_id}: {other_nodes_flow}")
                    finally:
                        shutil.rmtree(temp_dir, ignore_errors=True)

                    node_ids = list(all_other_nodes_flow[next(iter(all_other_nodes_flow))].keys())
                    df_combined = pd.DataFrame(node_ids, columns=['Node ID'])

                    for node_id, flows in all_other_nodes_flow.items():
                        flow_values = list(flows.values())
                        df_combined[f'{rainfall} Node {node_id} Peak Flow at {peak_times[node_id].strftime("%H:%M:%S")}'] = flow_values

                    output_csv_path = (
                        config['OUTPUT_CSV_BASE_PATH']
                        + f'{rainfall}_impervious_{imperviousness_para2}_rate_{minimum_rate}.csv'
                    )
                    output_dir = os.path.dirname(output_csv_path)
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)

                    df_combined.to_csv(output_csv_path, index=False, header=True)
                    print(
                        f"Results for {rainfall} (Imperviousness: {imperviousness_para2}, Minimum Rate: {minimum_rate}) "
                        f"have been saved to {output_csv_path}"
                    )
    if input("Merge all CSV files? (y/n, default: n): ").strip().lower() == 'y':
        csv_directory = get_user_input(
            "Enter the folder path containing all CSV files to merge, press Enter to skip",
            config['CSV_DIRECTORY']
        )
        output_combined_csv_path = combine_csv_files(csv_directory, config['CURRENT_DIR'])
        print("Merge complete. File saved to", output_combined_csv_path)
    else:
        print("Merge skipped.")


def get_flows_on_all_nodes_at_all_times(simulation_path, rainfall_data, imperviousness_data, minimum_rate_data, config):
    base_inp = SwmmInput.read_file(simulation_path)

    if sections.TIMESERIES in base_inp:
        print(f"Available time series in model: {list(base_inp[sections.TIMESERIES].keys())}")
    else:
        print("Warning: No time series defined in the model")

    if sections.RAINGAGES in base_inp:
        for rg_name, rg_data in base_inp[sections.RAINGAGES].items():
            print(f"Rain gauge {rg_name} settings: {rg_data}")

    available_timeseries = list(base_inp[sections.TIMESERIES].keys()) if sections.TIMESERIES in base_inp else []
    for rainfall in rainfall_data:
        if rainfall not in available_timeseries:
            print(f"Warning: Rainfall event '{rainfall}' has no corresponding time series in the model")

    for raingage_name in base_inp[sections.RAINGAGES]:
        for rainfall in rainfall_data:
            for imperviousness_para2 in imperviousness_data:
                for minimum_rate in minimum_rate_data:
                    temp_dir = tempfile.mkdtemp()
                    temp_inp_path = os.path.join(temp_dir, 'temp_model.inp')
                    try:
                        inp_copy = base_inp.copy()
                        inp_copy[sections.RAINGAGES][raingage_name]['timeseries'] = rainfall
                        form = 'CUMULATIVE' if rainfall in config['RAINFALL_EVENTS_SYNTHETIC'] else 'VOLUME'
                        interval = '0:06' if rainfall in config['RAINFALL_EVENTS_SYNTHETIC'] else '0:05'
                        inp_copy[sections.RAINGAGES][raingage_name]['interval'] = interval
                        inp_copy[sections.RAINGAGES][raingage_name]['form'] = form

                        for subcatchment_name in inp_copy[sections.SUBCATCHMENTS]:
                            inp_copy[sections.SUBCATCHMENTS][subcatchment_name]['imperviousness'] = imperviousness_para2
                        for infiltration_name in inp_copy[sections.INFILTRATION]:
                            inp_copy[sections.INFILTRATION][infiltration_name]['rate_min'] = minimum_rate

                        inp_copy[sections.OPTIONS]['REPORT_STEP'] = '00:05:00'
                        inp_copy.write_file(temp_inp_path)
                        print(f"Temporary model file saved: {temp_inp_path}")

                        all_nodes_flows = {}
                        print(
                            f"Starting simulation: Rainfall={rainfall}, Imperviousness={imperviousness_para2}, "
                            f"Min. Infiltration Rate={minimum_rate}, Interval={interval}, Form={form}"
                        )
                        with Simulation(temp_inp_path) as sim:
                            sim.step_advance(300)
                            nodes = Nodes(sim)
                            node_ids = [node.nodeid for node in nodes]
                            start_time_str = format_elapsed_sim_time(sim.start_time, sim.start_time)
                            flows_t0 = {node_id: nodes[node_id].total_inflow for node_id in node_ids}
                            all_nodes_flows[start_time_str] = flows_t0

                            step_count = 0
                            for _ in sim:
                                step_count += 1
                                current_time = sim.current_time
                                current_time_str = format_elapsed_sim_time(sim.start_time, current_time)
                                flows = {node_id: nodes[node_id].total_inflow for node_id in node_ids}
                                all_nodes_flows[current_time_str] = flows

                                if step_count <= 3:
                                    print(f"  Time: {current_time_str}, Total flow: {sum(flows.values()):.6f}")
                                if step_count % 100 == 0:
                                    print(
                                        f"  Progress - Time: {current_time_str}, Total flow: {sum(flows.values()):.6f}, Step: {step_count}"
                                    )
                        print(f"Simulation complete. Total steps: {step_count} steps")

                        total_flows = [sum(flows.values()) for flows in all_nodes_flows.values()]
                        max_total_flow = max(total_flows) if total_flows else 0
                        if max_total_flow == 0:
                            print(f"Warning: {rainfall} simulation produced no flow. Check rainfall data and model settings.")
                        else:
                            print(f"Maximum total flow: {max_total_flow:.6f}")

                        if all_nodes_flows:
                            node_ids = list(all_nodes_flows[next(iter(all_nodes_flows))].keys())
                            df_combined = pd.DataFrame(node_ids, columns=['Node ID'])
                            new_columns = []
                            for time_str, flows in all_nodes_flows.items():
                                flow_values = [flows.get(node_id, 0) for node_id in node_ids]
                                new_columns.append(pd.Series(flow_values, name=f'{rainfall} Flow Rate at {time_str}'))

                            df_combined = pd.concat([df_combined] + new_columns, axis=1)
                            output_csv_path = (
                                config['OUTPUT_CSV_BASE_PATH']
                                + f'{rainfall}_impervious_{imperviousness_para2}_rate_{minimum_rate}_all_times.csv'
                            )
                            output_dir = os.path.dirname(output_csv_path)
                            if not os.path.exists(output_dir):
                                os.makedirs(output_dir)
                            df_combined.to_csv(output_csv_path, index=False, header=True)
                            print(
                                f"Results saved: {rainfall} (Imperviousness: {imperviousness_para2}, "
                                f"Min. Infiltration Rate: {minimum_rate}) -> {output_csv_path}"
                            )
                        else:
                            print("Warning: No flow data was obtained.")
                    except Exception as e:
                        print(f"Error during simulation: {str(e)}")
                        traceback.print_exc()
                    finally:
                        try:
                            shutil.rmtree(temp_dir)
                            print(f"Temporary files cleaned up: {temp_dir}")
                        except Exception as e:
                            print(f"Error cleaning up temporary files: {str(e)}")

    if input("Merge all CSV files? (y/n, default: n): ").strip().lower() == 'y':
        csv_directory = get_user_input(
            "Enter the folder path containing all CSV files to merge, press Enter to skip",
            config['CSV_DIRECTORY']
        )
        combine_csv_files(csv_directory, config['CURRENT_DIR'])


def input_parameters(config, need_target_nodes=True):
    config['SIMULATION_PATH'] = prompt_for_existing_swmm_model(config.get('SIMULATION_PATH'))
    config['OUTPUT_CSV_BASE_PATH'] = normalize_user_path(config.get('OUTPUT_CSV_BASE_PATH')) or os.path.join(
        config['CURRENT_DIR'], '..', 'SWMM_Model', 'csv_files', 'node_flow_'
    )
    config['CSV_DIRECTORY'] = normalize_user_path(config.get('CSV_DIRECTORY')) or os.path.dirname(
        config['OUTPUT_CSV_BASE_PATH']
    )
    if not config.get('RAINFALL_EVENTS'):
        config['RAINFALL_EVENTS'] = list(config['RAINFALL_EVENTS_TRAINING'])

    if input("Change default parameters? (y/n, default: n): ").strip().lower() == 'y':
        config['SIMULATION_PATH'] = prompt_for_existing_swmm_model(
            get_user_input("Enter the SWMM model .inp file path, press Enter to skip", config['SIMULATION_PATH'])
        )
        config['OUTPUT_CSV_BASE_PATH'] = normalize_user_path(
            get_user_input("Enter the CSV output path prefix, press Enter to skip", config['OUTPUT_CSV_BASE_PATH'])
        )
        config['CSV_DIRECTORY'] = os.path.dirname(config['OUTPUT_CSV_BASE_PATH'])

        if need_target_nodes:
            config['TARGET_NODE_IDS'] = get_user_input(
                "Enter target node ID list (comma-separated), press Enter to skip",
                ','.join(config['TARGET_NODE_IDS'])
            ).split(',')
        config['RAINFALL_EVENTS'] = get_user_input(
            "Enter rainfall event list (comma-separated), press Enter to skip",
            ','.join(config['RAINFALL_EVENTS_TRAINING'])
        ).split(',')
        config['IMPERVIOUSNESS'] = get_user_input(
            "Enter imperviousness list (comma-separated), press Enter to skip",
            ','.join(config['IMPERVIOUSNESS'])
        ).split(',')
        config['MINIMUM_RATE'] = get_user_input(
            "Enter minimum infiltration rate list (comma-separated), press Enter to skip",
            ','.join(config['MINIMUM_RATE'])
        ).split(',')

    print("\n================================ Final Model Parameters: ==================================")
    print(f"Simulation file path: {config['SIMULATION_PATH']}")
    print(f"Output CSV path: {config['OUTPUT_CSV_BASE_PATH']}")
    if need_target_nodes:
        print(f"Target node ID list: {config['TARGET_NODE_IDS']}")
    print(f"Rainfall event list: {config['RAINFALL_EVENTS']}")
    print(f"Imperviousness list: {config['IMPERVIOUSNESS']}")
    print(f"Minimum infiltration rate list: {config['MINIMUM_RATE']}")
    print("==================================================================================")
    return config
