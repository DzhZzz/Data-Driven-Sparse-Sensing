import math
import os
import re
from pathlib import Path
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.linalg import qr

from DSS_analysis_utils import (
    _overall_event_nse,
    _reconstruct_from_measurements,
    _selected_sensor_measurements,
    _selected_sensors_for_r,
    Calculate_dropout_diagnostics,
    Calculate_dropout_delta_NSE,
    Calculate_eventwise_overall_nse_for_r,
    Calculate_eventwise_overall_nse_for_sensors,
    Calculate_eventwise_node_nse_details_for_r,
    Calculate_eventwise_overall_nse_noisy,
    Calculate_eventwise_nse_for_r,
    Calculate_eventwise_nse_noisy,
    Calculate_eventwise_nse_by_r,
    NSE,
    SVDQR,
    SVDQR_NSE,
    SVDQR_NSE_Exhaustive,
    SVDQR_NSE_Noisy,
    SVDQR_NSE_Random,
    SVDQR_NSE_Random_Eventwise,
    SVDQR_NSE_Random_Eventwise_Overall,
    SVDQR_NSE_SD,
    SVDQR_NSE_Nodewise,
    SVDQR_NSE_Exhaustive_Eventwise_Overall,
)
from DSS_plot_utils import (
    create_benchmark_paper_composite,
    generate_dropout_combined_delta_summary,
    generate_dropout_multifactor_analysis,
    plot_boxplot,
    plot_boxplot_Noisy,
    plot_noisy_dual_boxplot,
    plot_noisy_system_boxplot,
    plot_event_nse_linechart,
    plot_System_Level_Reconstruction_combined_boxplot_scatter,
    plot_System_Level_Reconstruction_combined_figure,
    plot_boxplot_random,
    plot_exhaustive_eventwise_benchmark,
    plot_exhaustive_eventwise_gap_only,
    plot_boxplot_SD,
    plot_cumulative_sum,
    plot_hydrograph_with_rainfall,
    plot_normalization_performance_comparison,
    plot_random_monte_carlo_eventwise_benchmark,
    plot_random_sampling_convergence_summary,
    plot_psi_heatmap,
    plot_residuals_heatmap,
    plot_shadowline,
    run_sensor_stability_analysis,
)
from DSS_swmm_utils import (
    combine_csv_files,
    get_flows_at_target_time,
    get_flows_on_all_nodes_at_all_times,
    get_flows_on_target_nodes_at_peak_time,
    get_model_info,
    get_user_input,
    input_parameters,
    prompt_for_existing_swmm_model,
)
OVERFLOW_NODES = {'J122', 'J110', 'OF-03'}

OPTION21_EVENT_FILES = [
    'Flowrates_6-2_2024.csv',
    'Flowrates_6-11_2024.csv',
    'Flowrates_6-28_2024.csv',
    'Flowrates_9-5_2019.csv',
    'Flowrates_9-17_2019.csv',
    'Flowrates_200-year.csv'
]


MENU_ITEMS = [
    ("1", "Get flow rates at peak time of target nodes"),
    ("2", "Get flow rates at all nodes at a target time"),
    ("3", "Get flow rates at all nodes for all time steps"),
    ("4", "Get model system parameters"),
    ("5", "Merge CSV files"),
    ("6", "Perform SVD and QR decomposition for optimal sensor placement"),
    ("7", "Plot cumulative sum of singular values for r"),
    ("8", "Plot confidence interval hydrograph"),
    ("9", "Plot NSE Boxplot (Node-level NSE)"),
    ("10", "Generate System-level NSE summary plots"),
    ("11", "Ambient noise analysis"),
    ("12", "Overall distribution of sensor dropouts (node + system level)"),
    ("13", "Sensor dropout analysis"),
    ("14", "Sensor Placement Benchmark Analysis"),
    ("15", "Plot Psi heatmap"),
    ("16", "Plot QR sensor diagnostics and model-contribution heatmaps"),
    ("17", "Plot hydrograph (rainfall + flow)"),
    ("18", "Single-r Exhaustive Search"),
    ("19", "Generate per-event reconstruction diagnostics"),
    ("20", "Sensor stability and cumulative energy analysis"),
    ("21", "Normalization Comparison Analysis"),
    ("22", "Random Sampling Convergence Analysis"),
    ("23", "Exit"),
]


# ---------------------------------------------------------------------------
# Menu / state helpers
# ---------------------------------------------------------------------------
def print_menu():
    print("\nSelect an operation:")
    for option, label in MENU_ITEMS:
        print(f"{option}. {label}")


def _state(state, key):
    return state[key]


def _current_dir(state):
    return _state(state, 'CURRENT_DIR')


def _nodewise_nse_dataframe(details):
    rows = []
    for node_id in details['ordered_node_ids']:
        row = {'Node_ID': node_id}
        for event_name, node_map in details['event_node_maps'].items():
            row[event_name] = node_map.get(node_id, np.nan)
        row['Mean_Across_Events'] = details['mean_node_map'].get(node_id, np.nan)
        rows.append(row)
    return pd.DataFrame(rows)


def _export_option10_nodewise_nse_csvs(state, real_event_matrices, design_event_matrices, r_values):
    output_root = os.path.join(_current_dir(state), 'Node_Level_NSE_CSV')
    os.makedirs(output_root, exist_ok=True)
    generated_files = []
    observed_mean_by_r = {}
    design_mean_by_r = {}
    combined_mean_by_r = {}
    grand_mean_values = {}

    for r in r_values:
        r_dir = os.path.join(output_root, f'r_{r}')
        os.makedirs(r_dir, exist_ok=True)

        observed_details = Calculate_eventwise_node_nse_details_for_r(
            real_event_matrices,
            _state(state, 'Node_ID'),
            r,
            _state(state, 'Psi'),
            _analysis_context(state)
        )
        observed_df = _nodewise_nse_dataframe(observed_details)
        observed_detail_path = os.path.join(r_dir, f'Node_Level_NSE_Per_Event_Observed_r{r}.csv')
        observed_mean_path = os.path.join(r_dir, f'Node_Level_NSE_Mean_Observed_r{r}.csv')
        observed_df.to_csv(observed_detail_path, index=False)
        observed_df[['Node_ID', 'Mean_Across_Events']].to_csv(observed_mean_path, index=False)
        generated_files.extend([observed_detail_path, observed_mean_path])
        observed_mean_by_r[r] = observed_df[['Node_ID', 'Mean_Across_Events']].rename(
            columns={'Mean_Across_Events': f'r_{r}'}
        )
        combined_mean_source = observed_df[['Node_ID', 'Mean_Across_Events']].rename(
            columns={'Mean_Across_Events': 'Observed_Mean_Across_Events'}
        )

        if design_event_matrices:
            design_details = Calculate_eventwise_node_nse_details_for_r(
                design_event_matrices,
                _state(state, 'Node_ID'),
                r,
                _state(state, 'Psi'),
                _analysis_context(state)
            )
            design_df = _nodewise_nse_dataframe(design_details)
            design_detail_path = os.path.join(r_dir, f'Node_Level_NSE_Per_Event_Design_r{r}.csv')
            design_mean_path = os.path.join(r_dir, f'Node_Level_NSE_Mean_Design_r{r}.csv')
            design_df.to_csv(design_detail_path, index=False)
            design_df[['Node_ID', 'Mean_Across_Events']].to_csv(design_mean_path, index=False)
            generated_files.extend([design_detail_path, design_mean_path])
            design_mean_by_r[r] = design_df[['Node_ID', 'Mean_Across_Events']].rename(
                columns={'Mean_Across_Events': f'r_{r}'}
            )
            combined_mean_source = combined_mean_source.merge(
                design_df[['Node_ID', 'Mean_Across_Events']].rename(
                    columns={'Mean_Across_Events': 'Design_Mean_Across_Events'}
                ),
                on='Node_ID',
                how='outer'
            )
            combined_mean_source[f'r_{r}'] = combined_mean_source[
                ['Observed_Mean_Across_Events', 'Design_Mean_Across_Events']
            ].mean(axis=1, skipna=True)
        else:
            combined_mean_source[f'r_{r}'] = combined_mean_source['Observed_Mean_Across_Events']

        combined_mean_path = os.path.join(r_dir, f'Node_Level_NSE_Mean_All_Events_r{r}.csv')
        combined_mean_source[['Node_ID', f'r_{r}']].rename(
            columns={f'r_{r}': 'Mean_All_Events'}
        ).to_csv(combined_mean_path, index=False)
        generated_files.append(combined_mean_path)
        combined_mean_by_r[r] = combined_mean_source[['Node_ID', f'r_{r}']]
        for _, row in combined_mean_source[['Node_ID', f'r_{r}']].iterrows():
            if pd.notna(row[f'r_{r}']):
                grand_mean_values.setdefault(row['Node_ID'], []).append(float(row[f'r_{r}']))

    if observed_mean_by_r:
        observed_summary = None
        for r in sorted(observed_mean_by_r):
            observed_summary = observed_mean_by_r[r] if observed_summary is None else observed_summary.merge(
                observed_mean_by_r[r],
                on='Node_ID',
                how='outer'
            )
        observed_summary_path = os.path.join(output_root, 'Node_Level_NSE_Mean_Observed_All_r.csv')
        observed_summary.to_csv(observed_summary_path, index=False)
        generated_files.append(observed_summary_path)

    if design_mean_by_r:
        design_summary = None
        for r in sorted(design_mean_by_r):
            design_summary = design_mean_by_r[r] if design_summary is None else design_summary.merge(
                design_mean_by_r[r],
                on='Node_ID',
                how='outer'
            )
        design_summary_path = os.path.join(output_root, 'Node_Level_NSE_Mean_Design_All_r.csv')
        design_summary.to_csv(design_summary_path, index=False)
        generated_files.append(design_summary_path)

    if combined_mean_by_r:
        combined_summary = None
        for r in sorted(combined_mean_by_r):
            combined_summary = combined_mean_by_r[r] if combined_summary is None else combined_summary.merge(
                combined_mean_by_r[r],
                on='Node_ID',
                how='outer'
            )
        combined_summary_path = os.path.join(output_root, 'Node_Level_NSE_Mean_All_Events_All_r.csv')
        combined_summary.to_csv(combined_summary_path, index=False)
        generated_files.append(combined_summary_path)

    if grand_mean_values:
        grand_mean_df = pd.DataFrame(
            [
                {
                    'Node_ID': node_id,
                    'Mean_All_Events_All_r': float(np.mean(values)),
                }
                for node_id, values in grand_mean_values.items()
            ]
        ).sort_values('Node_ID')
        grand_mean_path = os.path.join(output_root, 'Node_Level_NSE_Grand_Mean_All_Events_All_r.csv')
        grand_mean_df.to_csv(grand_mean_path, index=False)
        generated_files.append(grand_mean_path)

    return output_root, generated_files


def _analysis_context(state):
    return _state(state, 'analysis_context')


def _model_directory(state):
    return _state(state, 'MODEL_DIRECTORY')


def _swmm_config(state):
    return _state(state, 'swmm_config')


def _set_swmm_config(state, config):
    state['swmm_config'] = config


def _nse_node_ids(node_ids, data_matrix):
    return [
        node_id for node_id, node_series in zip(node_ids, data_matrix)
        if np.std(node_series) >= 0.01 or node_id in OVERFLOW_NODES
    ]


# ---------------------------------------------------------------------------
# Shared persistence / cache helpers
# ---------------------------------------------------------------------------
def _save_System_Level_Reconstruction_nse_tables(current_dir, filename, node_ids, by_r_values):
    rows = []
    for r, values in by_r_values.items():
        for node_id, value in zip(node_ids, values):
            rows.append({
                'Sensor_Count': r,
                'Node_ID': node_id,
                'NSE': value,
            })
    output_path = os.path.join(current_dir, filename)
    pd.DataFrame(rows).to_csv(output_path, index=False)
    return output_path


def _save_System_Level_Reconstruction_event_nse_tables(current_dir, filename, event_matrices, by_r_values):
    rows = []
    for r, values in by_r_values.items():
        for event_item, value in zip(event_matrices, values):
            rows.append({
                'Sensor_Count': r,
                'Event_Name': event_item['event_name'],
                'Overall_NSE': value,
            })
    output_path = os.path.join(current_dir, filename)
    pd.DataFrame(rows).to_csv(output_path, index=False)
    return output_path


def _load_System_Level_Reconstruction_scatter_truth_matrix(current_dir):
    testing_dir = os.path.join(current_dir, '..', 'Testing', 'Flowrates_data')
    event_parts = []
    for event_file in OPTION21_EVENT_FILES:
        event_path = os.path.join(testing_dir, event_file)
        if not os.path.exists(event_path):
            continue
        event_df = pd.read_csv(event_path)
        event_parts.append(event_df.iloc[:, 1:].to_numpy())
    if not event_parts:
        raise FileNotFoundError("No event files found for Option 22 scatter plot generation.")
    return np.concatenate(event_parts, axis=1)


def _distribution_summary_rows(level_label, by_r_values, scenario_label):
    rows = []
    for r, values in by_r_values.items():
        arr = np.asarray(values, dtype=float)
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            continue
        rows.append({
            'level': level_label,
            'r': r,
            'scenario': scenario_label,
            'q1': float(np.percentile(finite, 25)),
            'median': float(np.percentile(finite, 50)),
            'q3': float(np.percentile(finite, 75)),
            'mean': float(np.mean(finite)),
            'min': float(np.min(finite)),
            'max': float(np.max(finite)),
            'count': int(finite.size),
        })
    return rows


def _save_noise_boxplot_summary(
    current_dir,
    normalization_mode,
    num_monte_carlo,
    node_clean_by_r,
    node_noisy_5_by_r,
    node_noisy_10_by_r,
    node_noisy_15_by_r,
    system_clean_by_r,
    system_noisy_5_by_r,
    system_noisy_10_by_r,
    system_noisy_15_by_r,
):
    rows = []
    rows.extend(_distribution_summary_rows('node', node_clean_by_r, 'clean'))
    rows.extend(_distribution_summary_rows('node', node_noisy_5_by_r, '5%'))
    rows.extend(_distribution_summary_rows('node', node_noisy_10_by_r, '10%'))
    rows.extend(_distribution_summary_rows('node', node_noisy_15_by_r, '15%'))
    rows.extend(_distribution_summary_rows('system', system_clean_by_r, 'clean'))
    rows.extend(_distribution_summary_rows('system', system_noisy_5_by_r, '5%'))
    rows.extend(_distribution_summary_rows('system', system_noisy_10_by_r, '10%'))
    rows.extend(_distribution_summary_rows('system', system_noisy_15_by_r, '15%'))

    output_path = os.path.join(
        current_dir,
        f'noise_boxplot_stats_{normalization_mode}_r1_r10_mc{num_monte_carlo}.csv'
    )
    pd.DataFrame(rows).sort_values(['level', 'r', 'scenario']).to_csv(output_path, index=False)
    return output_path


def _system_level_noise_cache_path(state, normalization_mode, max_r, num_monte_carlo):
    filename = f'noise_system_level_distributions_{normalization_mode}_r1_r{max_r}_mc{num_monte_carlo}.npz'
    return _cached_npz_path(state, filename)


def _single_r_exhaustive_cache_path(state, r):
    filename = f'exhaustive_search_r{int(r)}_{_state(state, "normalization_mode")}.npz'
    return _cached_npz_path(state, filename)


def _analysis_utils_path(state):
    return os.path.join(_current_dir(state), 'DSS_analysis_utils.py')


def _paths_missing_or_incomplete(state, paths, reference_path=None):
    normalized_paths = [str(path) for path in paths if path is not None]
    if not normalized_paths:
        return True
    return any(not os.path.exists(path) for path in normalized_paths)


def _cache_status(state, cache_path, reference_path=None):
    exists = os.path.exists(cache_path)
    missing_or_incomplete = _paths_missing_or_incomplete(state, [cache_path], reference_path=reference_path) if exists else True
    return exists, missing_or_incomplete


def _prompt_use_existing_cache(state, cache_path, prompt, default='y', missing_message=None, reference_path=None):
    exists, missing_or_incomplete = _cache_status(state, cache_path, reference_path=reference_path)
    if exists and missing_or_incomplete:
        message = missing_message or f"Existing cache is incomplete and will not be used: {cache_path}"
        print(message)
        return False, exists, missing_or_incomplete
    if exists and not missing_or_incomplete:
        use_cache = get_user_input(prompt, default).lower() == 'y'
        return use_cache, exists, missing_or_incomplete
    return False, exists, missing_or_incomplete


def _prompt_recompute_cached_results(state, cache_path, recompute_prompt, default='n', missing_message=None, reference_path=None):
    exists, missing_or_incomplete = _cache_status(state, cache_path, reference_path=reference_path)
    if exists and missing_or_incomplete:
        message = missing_message or f"Existing cache is incomplete and will not be used: {cache_path}"
        print(message)
        return True, exists, missing_or_incomplete
    if exists and not missing_or_incomplete:
        should_recompute = get_user_input(recompute_prompt, default).lower() == 'y'
        return should_recompute, exists, missing_or_incomplete
    return True, exists, missing_or_incomplete


# ---------------------------------------------------------------------------
# Noise-analysis cache helpers
# ---------------------------------------------------------------------------
def _system_level_noise_cache_missing(state, cache_path):
    return _paths_missing_or_incomplete(state, [cache_path])


def _analysis_cache_missing(state, cache_path):
    return _paths_missing_or_incomplete(state, [cache_path])


def _default_exhaustive_workers():
    cpu_total = os.cpu_count() or 2
    return max(1, min(8, cpu_total // 2))


def _load_system_level_noise_cache(cache_path):
    saved = np.load(cache_path, allow_pickle=True)
    return {key: saved[key].item() for key in saved.files}


def _save_system_level_noise_cache(cache_path, clean_by_r, noisy_5_by_r, noisy_10_by_r, noisy_15_by_r):
    payload = {
        'clean': clean_by_r,
        '5%': noisy_5_by_r,
        '10%': noisy_10_by_r,
        '15%': noisy_15_by_r,
    }
    np.savez_compressed(cache_path, **payload)
    return cache_path


def _save_system_level_noise_summary(current_dir, normalization_mode, num_monte_carlo, cache_payload):
    rows = []
    rows.extend(_distribution_summary_rows('system', cache_payload['clean'], 'clean'))
    rows.extend(_distribution_summary_rows('system', cache_payload['5%'], '5%'))
    rows.extend(_distribution_summary_rows('system', cache_payload['10%'], '10%'))
    rows.extend(_distribution_summary_rows('system', cache_payload['15%'], '15%'))
    output_path = os.path.join(
        current_dir,
        f'noise_boxplot_stats_{normalization_mode}_r1_r10_mc{num_monte_carlo}.csv'
    )
    pd.DataFrame(rows).sort_values(['level', 'r', 'scenario']).to_csv(output_path, index=False)
    return output_path


# ---------------------------------------------------------------------------
# Dropout-analysis cache and output helpers
# ---------------------------------------------------------------------------
def _dropout_output_dir(state):
    return os.path.join(
        _current_dir(state),
        f"Dropout_Discussion_{_state(state, 'normalization_mode')}"
    )


def _dropout_combined_csv_path(state):
    return os.path.join(
        _current_dir(state),
        f'dropout_system_level_diagnostics_{_state(state, "normalization_mode")}_r2_r10.csv'
    )


def _dropout_legacy_combined_csv_path(state):
    old_level_name = 'system' + 'wise'
    return os.path.join(
        _current_dir(state),
        f'dropout_{old_level_name}_diagnostics_{_state(state, "normalization_mode")}_r2_r10.csv'
    )


def _dropout_per_r_path(output_dir, r):
    return os.path.join(output_dir, f'dropout_system_level_diagnostics_r{r}.csv')


def _dropout_legacy_per_r_path(output_dir, r):
    old_level_name = 'system' + 'wise'
    return os.path.join(output_dir, f'dropout_{old_level_name}_diagnostics_r{r}.csv')


def _dropout_diagnostics_missing(state, combined_path, per_r_paths):
    return _paths_missing_or_incomplete(state, [combined_path] + list(per_r_paths))


def _ensure_dropout_diagnostics_files(state, r_values=None, force_recompute=False):
    if r_values is None:
        r_values = range(2, 11)
    r_values = sorted({int(r) for r in r_values})
    output_dir = _dropout_output_dir(state)
    os.makedirs(output_dir, exist_ok=True)

    per_r_paths = [_dropout_per_r_path(output_dir, r) for r in r_values]
    combined_path = _dropout_combined_csv_path(state)
    legacy_per_r_paths = [_dropout_legacy_per_r_path(output_dir, r) for r in r_values]
    legacy_combined_path = _dropout_legacy_combined_csv_path(state)

    if not force_recompute and not _dropout_diagnostics_missing(state, combined_path, per_r_paths):
        return output_dir, combined_path, False
    if not force_recompute and not _dropout_diagnostics_missing(state, legacy_combined_path, legacy_per_r_paths):
        return output_dir, legacy_combined_path, False

    combined_rows = []
    for r, per_r_path in zip(r_values, per_r_paths):
        diagnostics = Calculate_dropout_diagnostics(
            _state(state, 'Node_ID'),
            r,
            _state(state, 'Psi'),
            _analysis_context(state),
            event_matrices=_state(state, 'testing_event_matrices'),
            system_level_event_nse=True,
        )
        df = pd.DataFrame(diagnostics)
        df.to_csv(per_r_path, index=False)
        df_with_r = df.copy()
        if 'r' in df_with_r.columns:
            df_with_r['r'] = r
            ordered_cols = ['r'] + [col for col in df_with_r.columns if col != 'r']
            df_with_r = df_with_r[ordered_cols]
        else:
            df_with_r.insert(0, 'r', r)
        combined_rows.append(df_with_r)

    if combined_rows:
        combined_df = pd.concat(combined_rows, ignore_index=True)
        combined_df.to_csv(combined_path, index=False)
        return output_dir, combined_path, True

    return output_dir, None, True


def _generate_dropout_analysis_bundle(state, r_values, ensure_diagnostics=True):
    if ensure_diagnostics:
        output_dir, combined_csv_path, regenerated = _ensure_dropout_diagnostics_files(state, r_values=range(2, 11))
    else:
        output_dir = _dropout_output_dir(state)
        combined_csv_path = _dropout_combined_csv_path(state)
        regenerated = False

    combined_figure_path = generate_dropout_combined_delta_summary(
        _state(state, 'Psi'),
        r_values=r_values,
        current_dir=_current_dir(state),
        normalization_mode=_state(state, 'normalization_mode'),
    )
    multifactor_outputs = generate_dropout_multifactor_analysis(
        _state(state, 'Psi'),
        current_dir=_current_dir(state),
        normalization_mode=_state(state, 'normalization_mode'),
    )

    return {
        'output_dir': output_dir,
        'combined_csv_path': combined_csv_path,
        'regenerated': regenerated,
        'combined_figure_path': combined_figure_path,
        'multifactor_outputs': multifactor_outputs,
    }


def _print_dropout_analysis_bundle(title, bundle):
    print(title)
    print(f"  Diagnostics directory: {bundle['output_dir']}")
    print(f"  Combined diagnostics CSV: {bundle['combined_csv_path']}")
    print(f"  Diagnostics source: {'recomputed' if bundle['regenerated'] else 'cached'}")
    print(f"  Combined delta summary: {bundle['combined_figure_path']}")
    print(f"  Multifactor summary: {bundle['multifactor_outputs']['summary']}")
    print(f"  Model contribution diagnostics: {bundle['multifactor_outputs']['pivot_vs_full_contribution_r2']}")
    print(f"  Multifactor coefficients: {bundle['multifactor_outputs']['coefficients']}")


# ---------------------------------------------------------------------------
# Event-based analysis helpers
# ---------------------------------------------------------------------------
def _load_first_real_event_matrix(current_dir):
    testing_dir = os.path.join(current_dir, '..', 'Testing', 'Flowrates_data')
    for event_file in OPTION21_EVENT_FILES:
        if '200-year' in event_file.lower():
            continue
        event_path = os.path.join(testing_dir, event_file)
        if not os.path.exists(event_path):
            continue
        event_df = pd.read_csv(event_path)
        return event_file, event_df.iloc[:, 1:].to_numpy()
    raise FileNotFoundError("No real rainfall event files found for Option 9.")


def _load_option9_event_matrices(current_dir):
    testing_dir = os.path.join(current_dir, '..', 'Testing', 'Flowrates_data')
    selected_events = []

    real_event_files = [event_file for event_file in OPTION21_EVENT_FILES if '200-year' not in event_file.lower()]
    for event_file in real_event_files[:2]:
        event_path = os.path.join(testing_dir, event_file)
        if not os.path.exists(event_path):
            continue
        event_df = pd.read_csv(event_path)
        selected_events.append({
            'event_file': event_file,
            'event_matrix': event_df.iloc[:, 1:].to_numpy(),
            'event_type': 'real',
            'time_labels': event_df.columns[1:].tolist(),
        })

    for event_file in OPTION21_EVENT_FILES:
        if '200-year' not in event_file.lower():
            continue
        event_path = os.path.join(testing_dir, event_file)
        if not os.path.exists(event_path):
            continue
        event_df = pd.read_csv(event_path)
        selected_events.append({
            'event_file': event_file,
            'event_matrix': event_df.iloc[:, 1:].to_numpy(),
            'event_type': 'design',
            'time_labels': event_df.columns[1:].tolist(),
        })
        break

    if len(selected_events) < 3:
        raise FileNotFoundError("Option 9 requires two real rainfall events and one 200-year event.")
    return selected_events


def _find_prefixed_file(directory, prefix, extension):
    if not os.path.isdir(directory):
        return None

    candidates = [
        filename for filename in os.listdir(directory)
        if filename.lower().endswith(extension.lower()) and filename.startswith(prefix)
    ]
    if not candidates and not prefix.startswith('Flowrates_') and extension.lower() == '.csv':
        candidates = [
            filename for filename in os.listdir(directory)
            if filename.lower().endswith('.csv') and filename.startswith(f'Flowrates_{prefix}')
        ]
    if not candidates and not prefix.startswith('Rainfall_') and extension.lower() in {'.xlsx', '.xls'}:
        candidates = [
            filename for filename in os.listdir(directory)
            if filename.lower().endswith(extension.lower()) and filename.startswith(f'Rainfall_{prefix}')
        ]
    return os.path.join(directory, sorted(candidates)[0]) if candidates else None


def _resolve_hydrograph_paths(current_dir, dataset_scope, event_name):
    dataset_key = str(dataset_scope).strip().lower()
    if dataset_key in {'1', 'training', 'train'}:
        flow_dir = os.path.join(current_dir, '..', 'Training', 'Flowrates_data')
        rainfall_dir = os.path.join(current_dir, '..', 'Training', 'Rainfall_data', 'Real_rainfall')
    else:
        flow_dir = os.path.join(current_dir, '..', 'Testing', 'Flowrates_data')
        if '200-year' in event_name.lower():
            rainfall_dir = os.path.join(current_dir, '..', 'Testing', 'Rainfall_data', 'Synthetic rainfall')
        else:
            rainfall_dir = os.path.join(current_dir, '..', 'Testing', 'Rainfall_data', 'Real rainfall')

    flow_path = _find_prefixed_file(flow_dir, event_name, '.csv')
    rainfall_path = _find_prefixed_file(rainfall_dir, event_name, '.xlsx')
    return flow_path, rainfall_path


def _mode_specific_context(state, normalization_mode):
    mode = str(normalization_mode).strip().lower()
    x_matrix = _state(state, 'X')
    x_max_limits = _state(state, 'X_max_limits')

    if mode in {'global_minmax', 'global_max'}:
        x_min = float(np.min(x_matrix))
        x_max = float(np.max(x_matrix))
        scale = x_max - x_min
        safe_scale = scale if scale > 0 else 1.0
        x_mean = np.full((x_matrix.shape[0], 1), x_min, dtype=float)
        x_std_safe = np.full((x_matrix.shape[0], 1), safe_scale, dtype=float)
        normalized_mode = 'global_minmax'
    else:
        x_mean = np.mean(x_matrix, axis=1, keepdims=True)
        x_std = np.std(x_matrix, axis=1, keepdims=True)
        x_std_safe = np.where(x_std == 0, 1.0, x_std)
        normalized_mode = 'zscore'

    return {
        'X_mean': x_mean,
        'X_std_safe': x_std_safe,
        'X_max_limits': x_max_limits,
        'Node_ID': _state(state, 'Node_ID'),
        'normalization_mode': normalized_mode,
    }


def _summarize_distribution(values):
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return {
            'count': 0,
            'mean': np.nan,
            'std': np.nan,
            'p05': np.nan,
            'q1': np.nan,
            'median': np.nan,
            'q3': np.nan,
            'p95': np.nan,
        }
    return {
        'count': int(finite.size),
        'mean': float(np.mean(finite)),
        'std': float(np.std(finite)),
        'p05': float(np.percentile(finite, 5)),
        'q1': float(np.percentile(finite, 25)),
        'median': float(np.percentile(finite, 50)),
        'q3': float(np.percentile(finite, 75)),
        'p95': float(np.percentile(finite, 95)),
    }


def _pooled_stat_delta(stats_a, stats_b):
    tracked = ['p05', 'q1', 'median', 'q3', 'p95']
    return max(abs(stats_a[key] - stats_b[key]) for key in tracked)


def _recommend_random_sample_size(r, combo_count, per_size_stats, reference_stats, tol_main, tol_tail):
    if combo_count <= min(max(per_size_stats.keys()), 10000):
        return 'exact'

    for sample_size in sorted(per_size_stats.keys()):
        seed_stats = per_size_stats[sample_size]
        main_ranges = []
        tail_ranges = []
        ref_deltas_main = []
        ref_deltas_tail = []

        for stat_name in ['q1', 'median', 'q3']:
            values = [seed_stats[seed][stat_name] for seed in seed_stats]
            main_ranges.append(max(values) - min(values))
        for stat_name in ['p05', 'p95']:
            values = [seed_stats[seed][stat_name] for seed in seed_stats]
            tail_ranges.append(max(values) - min(values))

        if reference_stats is not None:
            for seed in seed_stats:
                ref_deltas_main.append(max(
                    abs(seed_stats[seed]['q1'] - reference_stats[seed]['q1']),
                    abs(seed_stats[seed]['median'] - reference_stats[seed]['median']),
                    abs(seed_stats[seed]['q3'] - reference_stats[seed]['q3']),
                ))
                ref_deltas_tail.append(max(
                    abs(seed_stats[seed]['p05'] - reference_stats[seed]['p05']),
                    abs(seed_stats[seed]['p95'] - reference_stats[seed]['p95']),
                ))

        if (
            max(main_ranges) <= tol_main
            and max(tail_ranges) <= tol_tail
            and (reference_stats is None or max(ref_deltas_main) <= tol_main)
            and (reference_stats is None or max(ref_deltas_tail) <= tol_tail)
        ):
            return sample_size

    return f"> {max(per_size_stats.keys())}"


def _sensor_set_event_nse(sensor_indices, events, psi, r, context):
    values = []
    for event_item in events:
        event_data = event_item['data']
        reconstructed = _reconstruct_from_measurements(
            _selected_sensor_measurements(event_data, sensor_indices, context),
            sensor_indices,
            psi,
            r,
            context
        )
        values.append(float(_overall_event_nse(reconstructed, event_data, context['Node_ID'])))
    return values


def _exact_random_distribution(r, events, psi, context):
    values = []
    for combo in combinations(range(psi.shape[0]), r):
        values.extend(_sensor_set_event_nse(np.asarray(combo, dtype=int), events, psi, r, context))
    return np.asarray(values, dtype=float)


def _monte_carlo_random_distribution(r, sample_size, seed, events, psi, context):
    rng = np.random.default_rng(seed)
    all_indices = np.arange(psi.shape[0])
    values = []
    for _ in range(sample_size):
        sensor_indices = np.sort(rng.choice(all_indices, size=r, replace=False))
        values.extend(_sensor_set_event_nse(sensor_indices, events, psi, r, context))
    return np.asarray(values, dtype=float)


def _parse_axis_range(range_text):
    text = str(range_text).strip()
    if not text:
        return None
    parts = [part.strip() for part in text.split(',')]
    if len(parts) != 2:
        raise ValueError("Axis range must be provided as min,max")
    lower = float(parts[0])
    upper = float(parts[1])
    if upper <= lower:
        raise ValueError("Axis upper bound must be greater than lower bound")
    return lower, upper


def _option21_model_directory(current_dir):
    return os.path.join(current_dir, 'saved_models')


def _option21_svd_path_by_mode(current_dir, normalization_mode='zscore'):
    mode = str(normalization_mode).strip().lower()
    suffix = 'global_minmax' if mode in {'global_minmax', 'global_max'} else 'zscore'
    return os.path.join(_option21_model_directory(current_dir), f'svd_decomposition_{suffix}.npz')


def load_svd_decomposition(current_dir, X, X_mean, X_std_safe, normalization_mode='zscore'):
    model_directory = _option21_model_directory(current_dir)
    svd_save_path = _option21_svd_path_by_mode(current_dir, normalization_mode)
    legacy_svd_path = os.path.join(current_dir, 'svd_decomposition_standardized.npz')
    legacy_mode_svd_path = os.path.join(model_directory, 'svd_decomposition_global_max.npz')
    os.makedirs(model_directory, exist_ok=True)

    if normalization_mode == 'zscore' and not os.path.exists(svd_save_path) and os.path.exists(legacy_svd_path):
        os.replace(legacy_svd_path, svd_save_path)
        print(f"Moved legacy SVD file to dedicated model folder: {svd_save_path}")

    if (
        str(normalization_mode).strip().lower() in {'global_minmax', 'global_max'}
        and not os.path.exists(svd_save_path)
        and os.path.exists(legacy_mode_svd_path)
    ):
        os.replace(legacy_mode_svd_path, svd_save_path)
        print(f"Renamed legacy normalization SVD file to: {svd_save_path}")

    if os.path.exists(svd_save_path):
        svd_data = np.load(svd_save_path)
        print(f"SVD decomposition results ({svd_save_path}) loaded.")
        return svd_data['Psi'], svd_data['S'], svd_data['V']

    print("Computing SVD for event analysis...")
    X_normalized = (X - X_mean) / X_std_safe
    Psi_basis, singular_values, temporal_modes = np.linalg.svd(X_normalized, full_matrices=False)
    np.savez(svd_save_path, Psi=Psi_basis, S=singular_values, V=temporal_modes)
    print(f"SVD decomposition results saved to {svd_save_path}.")
    return Psi_basis, singular_values, temporal_modes


def _reconstruct_event_physical(data, r, Psi, X_mean, X_std_safe, X_max_limits):
    ind_Psi = np.arange(r)
    Psi_ind = Psi[:, ind_Psi]
    _, _, pivoting = qr(Psi_ind.T, pivoting=True)
    Pivot = pivoting[:r]
    Psi_pivot = Psi[Pivot, :][:, ind_Psi]
    Psi_pivot_pinv = np.linalg.pinv(Psi_pivot)

    data_normalized = (data - X_mean) / X_std_safe
    reconstructed_normalized = Psi_ind @ Psi_pivot_pinv @ data_normalized[Pivot]
    reconstructed_physical = reconstructed_normalized * X_std_safe + X_mean
    reconstructed_physical = np.maximum(reconstructed_physical, 0.0)
    return reconstructed_physical, Pivot


def _should_include_nse_node(true_series, node_id):
    normalized_node_id = re.sub(r'(OF)(\d+)', r'\1-\2', str(node_id))
    if normalized_node_id in {'J122', 'J110', 'OF-03'}:
        return True
    return np.std(true_series) >= 0.01


def _compute_nse_series(reconstructed_data, true_data, node_ids, nse_func):
    nse_values = []
    valid_indices = []
    for i in range(true_data.shape[0]):
        if not _should_include_nse_node(true_data[i, :], node_ids[i]):
            continue
        nse_values.append(nse_func(reconstructed_data[i, :], true_data[i, :]))
        valid_indices.append(i)
    return nse_values, valid_indices


def _plot_event_reconstruction_comparison(true_data, reconstructed_data, node_ids, r, event_name, output_dir, nse_func, log_prefix='Event reconstruction'):
    print(f"Generating {log_prefix} plot for event {event_name} (r={r})...")
    num_nodes = true_data.shape[0]
    cols = 8
    rows = (num_nodes + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3), dpi=100)
    axes = np.atleast_1d(axes).flatten()

    for i in range(num_nodes):
        ax = axes[i]
        true_val = true_data[i, :]
        reconstructed_val = reconstructed_data[i, :]
        line1, = ax.plot(true_val, label='True Observed', color='blue', alpha=0.65, linewidth=1.4)
        line2, = ax.plot(
            reconstructed_val,
            label='Reconstructed',
            color='red',
            linestyle='--',
            alpha=0.85,
            linewidth=1.2
        )
        node_nse = nse_func(reconstructed_val, true_val)
        ax.set_title(f"Node: {node_ids[i]}\nNSE: {node_nse:.2f}", fontsize=9)
        ax.tick_params(labelsize=7)

    for i in range(num_nodes, len(axes)):
        axes[i].axis('off')

    fig.legend(
        [line1, line2],
        ['True Observed Flow', 'Reconstructed Flow'],
        loc='upper center',
        bbox_to_anchor=(0.5, 0.99),
        ncol=2,
        fontsize=14,
        frameon=True,
        shadow=True
    )
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plot_path = os.path.join(output_dir, f'Event_Reconstruction_Comparison_{event_name}_r{r}.png')
    plt.savefig(plot_path, dpi=300)
    plt.close()
    return plot_path


def run_event_reconstruction_analysis(
    r_values,
    current_dir,
    X,
    X_mean,
    X_std_safe,
    X_max_limits,
    node_ids,
    nse_func,
    normalization_mode='zscore',
    output_root_name='Event_Reconstruction',
    log_prefix='Event reconstruction'
):
    testing_dir = os.path.join(current_dir, '..', 'Testing', 'Flowrates_data')
    output_root = os.path.join(current_dir, output_root_name)
    os.makedirs(output_root, exist_ok=True)

    Psi_basis, _, _ = load_svd_decomposition(current_dir, X, X_mean, X_std_safe, normalization_mode)
    summary_rows = []
    all_event_nse_frames = []
    all_merged_nse_frames = []

    for r in r_values:
        r_output_dir = os.path.join(output_root, f'r_{r}')
        os.makedirs(r_output_dir, exist_ok=True)

        merged_true_parts = []
        merged_reconstructed_parts = []
        merged_columns = []
        sensor_indices = None

        for event_file in OPTION21_EVENT_FILES:
            event_path = os.path.join(testing_dir, event_file)
            if not os.path.exists(event_path):
                print(f"Warning: event file not found and skipped: {event_path}")
                continue

            event_df = pd.read_csv(event_path)
            event_name = os.path.splitext(event_file)[0]
            event_columns = event_df.columns[1:].tolist()
            event_data = event_df.iloc[:, 1:].to_numpy()

            reconstructed_event, Pivot = _reconstruct_event_physical(
                event_data,
                r,
                Psi_basis,
                X_mean,
                X_std_safe,
                X_max_limits,
            )
            nse_values, valid_nse_indices = _compute_nse_series(
                reconstructed_event,
                event_data,
                node_ids,
                nse_func
            )
            sensor_indices = Pivot

            merged_true_parts.append(event_data)
            merged_reconstructed_parts.append(reconstructed_event)
            merged_columns.extend(event_columns)

            reconstructed_df = pd.DataFrame(reconstructed_event, columns=event_columns)
            reconstructed_df.insert(0, 'Node_ID', node_ids)
            reconstructed_output_path = os.path.join(r_output_dir, f'{event_name}_reconstructed_r{r}.csv')
            reconstructed_df.to_csv(reconstructed_output_path, index=False)

            nse_df = pd.DataFrame({
                'Sensor_Count': r,
                'Event': event_name,
                'Node_ID': [node_ids[idx] for idx in valid_nse_indices],
                'NSE': nse_values
            })
            nse_output_path = os.path.join(r_output_dir, f'{event_name}_nse_r{r}.csv')
            nse_df.to_csv(nse_output_path, index=False)
            all_event_nse_frames.append(nse_df)

            plot_output_path = _plot_event_reconstruction_comparison(
                event_data,
                reconstructed_event,
                node_ids,
                r,
                event_name,
                r_output_dir,
                nse_func,
                log_prefix=log_prefix
            )

            selected_sensor_ids = [node_ids[int(idx)] for idx in Pivot]
            summary_rows.append({
                'Sensor_Count': r,
                'Event': event_name,
                'Mean_NSE': float(np.mean(nse_values)),
                'Sensor_Indices': ','.join(map(str, Pivot)),
                'Sensor_Node_IDs': ','.join(selected_sensor_ids),
                'Reconstructed_CSV': reconstructed_output_path,
                'NSE_CSV': nse_output_path,
                'Plot_Path': plot_output_path
            })

        if not merged_true_parts:
            continue

        merged_true_data = np.concatenate(merged_true_parts, axis=1)
        merged_reconstructed_data = np.concatenate(merged_reconstructed_parts, axis=1)
        merged_nse_values, merged_valid_indices = _compute_nse_series(
            merged_reconstructed_data,
            merged_true_data,
            node_ids,
            nse_func
        )

        merged_true_df = pd.DataFrame(merged_true_data, columns=merged_columns)
        merged_true_df.insert(0, 'Node_ID', node_ids)
        merged_true_output_path = os.path.join(r_output_dir, f'All_Events_True_r{r}.csv')
        merged_true_df.to_csv(merged_true_output_path, index=False)

        merged_reconstructed_df = pd.DataFrame(merged_reconstructed_data, columns=merged_columns)
        merged_reconstructed_df.insert(0, 'Node_ID', node_ids)
        merged_reconstructed_output_path = os.path.join(r_output_dir, f'All_Events_Reconstructed_r{r}.csv')
        merged_reconstructed_df.to_csv(merged_reconstructed_output_path, index=False)

        merged_nse_df = pd.DataFrame({
            'Sensor_Count': r,
            'Event': 'All_Events_Merged',
            'Node_ID': [node_ids[idx] for idx in merged_valid_indices],
            'NSE': merged_nse_values
        })
        merged_nse_output_path = os.path.join(r_output_dir, f'All_Events_NSE_r{r}.csv')
        merged_nse_df.to_csv(merged_nse_output_path, index=False)
        all_merged_nse_frames.append(merged_nse_df)

        if sensor_indices is not None:
            selected_sensor_ids = [node_ids[int(idx)] for idx in sensor_indices]
            print(f"{log_prefix} completed for r={r}. Selected sensors: {selected_sensor_ids}")
        print(f"Merged true matrix saved to {merged_true_output_path}")
        print(f"Merged reconstructed matrix saved to {merged_reconstructed_output_path}")
        print(f"Merged NSE saved to {merged_nse_output_path}")

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows).sort_values(['Sensor_Count', 'Event'])
        summary_output_path = os.path.join(output_root, 'Event_Reconstruction_Summary.csv')
        summary_df.to_csv(summary_output_path, index=False)
        print(f"Event reconstruction summary saved to {summary_output_path}")

    if all_event_nse_frames:
        event_nse_output_path = os.path.join(output_root, 'Event_Reconstruction_Event_NSE_All.csv')
        pd.concat(all_event_nse_frames, ignore_index=True).sort_values(
            ['Sensor_Count', 'Event', 'Node_ID']
        ).to_csv(event_nse_output_path, index=False)
        print(f"All system-level NSE results saved to {event_nse_output_path}")

    if all_merged_nse_frames:
        merged_nse_output_path = os.path.join(output_root, 'Event_Reconstruction_Merged_NSE_By_R.csv')
        pd.concat(all_merged_nse_frames, ignore_index=True).sort_values(
            ['Sensor_Count', 'Node_ID']
        ).to_csv(merged_nse_output_path, index=False)
        print(f"Merged NSE grouped by sensor count saved to {merged_nse_output_path}")


# ---------------------------------------------------------------------------
# Saved-model cache path helpers
# ---------------------------------------------------------------------------
def _representative_r_values(r_values):
    ordered = list(r_values)
    if not ordered:
        return []
    candidates = [ordered[0], ordered[len(ordered) // 2], ordered[-1]]
    return list(dict.fromkeys(candidates))


def _noise_sample_path(state, dataset_tag, r, noise_level, num_monte_carlo):
    noise_tag = str(noise_level).replace('.', 'p')
    return os.path.join(
        _model_directory(state),
        f'ambient_noise_samples_{dataset_tag}_r{r}_noise_{noise_tag}_mc{num_monte_carlo}.npz'
    )


def _legacy_noise_sample_path(state, dataset_tag, r, noise_level, num_monte_carlo):
    noise_tag = str(noise_level).replace('.', 'p')
    return os.path.join(
        _current_dir(state),
        f'ambient_noise_samples_{dataset_tag}_r{r}_noise_{noise_tag}_mc{num_monte_carlo}.npz'
    )


def _prepare_noise_sample_path(state, dataset_tag, r, noise_level, num_monte_carlo):
    os.makedirs(_model_directory(state), exist_ok=True)
    target_path = _noise_sample_path(state, dataset_tag, r, noise_level, num_monte_carlo)
    legacy_path = _legacy_noise_sample_path(state, dataset_tag, r, noise_level, num_monte_carlo)

    # Backward compatibility for caches created before dataset-specific tagging.
    if dataset_tag == 'real':
        generic_legacy_path = os.path.join(
            _current_dir(state),
            f"ambient_noise_samples_r{r}_noise_{str(noise_level).replace('.', 'p')}_mc{num_monte_carlo}.npz"
        )
    else:
        generic_legacy_path = None

    if os.path.exists(target_path):
        return target_path

    if os.path.exists(legacy_path):
        os.replace(legacy_path, target_path)
        print(f"Moved legacy noise samples to dedicated model folder: {target_path}")
    elif generic_legacy_path and os.path.exists(generic_legacy_path):
        os.replace(generic_legacy_path, target_path)
        print(f"Moved legacy noise samples to dedicated model folder: {target_path}")

    return target_path


def _cached_npz_path(state, filename):
    os.makedirs(_model_directory(state), exist_ok=True)
    target_path = os.path.join(_model_directory(state), filename)
    legacy_path = os.path.join(_current_dir(state), filename)
    legacy_mode_filename = filename.replace('global_minmax', 'global_max')
    legacy_mode_target_path = os.path.join(_model_directory(state), legacy_mode_filename)
    legacy_mode_current_path = os.path.join(_current_dir(state), legacy_mode_filename)

    if os.path.exists(target_path):
        return target_path

    if legacy_mode_filename != filename and os.path.exists(legacy_mode_target_path):
        os.replace(legacy_mode_target_path, target_path)
        print(f"Renamed legacy cache file to: {target_path}")
        return target_path

    if legacy_mode_filename != filename and os.path.exists(legacy_mode_current_path):
        os.replace(legacy_mode_current_path, target_path)
        print(f"Moved legacy cache file to dedicated model folder: {target_path}")
        return target_path

    if os.path.exists(legacy_path):
        os.replace(legacy_path, target_path)
        print(f"Moved legacy cache file to dedicated model folder: {target_path}")

    return target_path


def _default_random_mc_iterations(state, r):
    mode = str(_state(state, 'normalization_mode')).strip().lower()
    if mode in {'global_minmax', 'global_max'}:
        return 20000
    if r >= 5:
        return 10000
    return math.comb(len(_state(state, 'Node_ID')), r)


def _load_or_create_noise_samples(state, dataset_tag, data_matrix, r, noise_level, num_monte_carlo):
    path = _prepare_noise_sample_path(state, dataset_tag, r, noise_level, num_monte_carlo)
    if os.path.exists(path):
        saved = np.load(path)
        noise_samples = saved['noise_samples']
        expected_shape = (num_monte_carlo, r, data_matrix.shape[1])
        if noise_samples.shape == expected_shape:
            print(f"Loaded noise samples from {path}")
            return noise_samples
        print(f"Cached noise samples shape mismatch for {dataset_tag}; regenerating {path}")

    noise_samples = np.random.normal(0, noise_level / 3.0, size=(num_monte_carlo, r, data_matrix.shape[1]))
    np.savez_compressed(path, noise_samples=noise_samples)
    print(f"Saved noise samples to {path}")
    return noise_samples


# ---------------------------------------------------------------------------
# Menu handlers
# ---------------------------------------------------------------------------
def handle_option_1(state):
    swmm_config = input_parameters(_swmm_config(state), need_target_nodes=True)
    _set_swmm_config(state, swmm_config)
    get_flows_on_target_nodes_at_peak_time(
        swmm_config['SIMULATION_PATH'],
        swmm_config['TARGET_NODE_IDS'],
        swmm_config['RAINFALL_EVENTS'],
        swmm_config['IMPERVIOUSNESS'],
        swmm_config['MINIMUM_RATE'],
        swmm_config
    )
    return True


def handle_option_2(state):
    swmm_config = input_parameters(_swmm_config(state), need_target_nodes=False)
    _set_swmm_config(state, swmm_config)
    get_flows_at_target_time(
        swmm_config['SIMULATION_PATH'],
        swmm_config['RAINFALL_EVENTS'],
        swmm_config['IMPERVIOUSNESS'],
        swmm_config['MINIMUM_RATE'],
        swmm_config
    )
    return True


def handle_option_3(state):
    swmm_config = input_parameters(_swmm_config(state), need_target_nodes=False)
    _set_swmm_config(state, swmm_config)
    get_flows_on_all_nodes_at_all_times(
        swmm_config['SIMULATION_PATH'],
        swmm_config['RAINFALL_EVENTS'],
        swmm_config['IMPERVIOUSNESS'],
        swmm_config['MINIMUM_RATE'],
        swmm_config
    )
    return True


def handle_option_4(state):
    print("============================== Model System Parameters: ==============================")
    swmm_config = _swmm_config(state)
    swmm_config['SIMULATION_PATH'] = prompt_for_existing_swmm_model(swmm_config.get('SIMULATION_PATH'))
    _set_swmm_config(state, swmm_config)
    get_model_info(swmm_config['SIMULATION_PATH'])
    print("===========================================================================")
    return True


def handle_option_5(state):
    csv_directory = get_user_input(
        "Enter the folder path containing all CSV files to merge, press Enter to skip",
        _swmm_config(state)['CSV_DIRECTORY']
    )
    output_combined_csv_path = combine_csv_files(csv_directory, _current_dir(state))
    print("Merge complete. File saved to", output_combined_csv_path)
    return True


def handle_option_6(state):
    r = int(get_user_input("Enter r value, press Enter to skip", 3))
    Pivot, residuals, relative_residuals = SVDQR(_state(state, 'X'), _state(state, 'Psi'), _state(state, 'S'), r)
    print(f"\n========== Optimal Sensor Placement Analysis (r={r}) ==========")
    print(f"Selected sensor indices (Pivot): {Pivot}")
    print(f"\nDetails:")
    for i, (pivot_idx, res, rel_res) in enumerate(zip(Pivot, residuals, relative_residuals)):
        pivot_idx_int = int(pivot_idx)
        node_id_raw = _state(state, 'Node_ID_raw')[pivot_idx_int] if pivot_idx_int < len(_state(state, 'Node_ID_raw')) else "Unknown"
        node_id_formatted = _state(state, 'Node_ID')[pivot_idx_int] if pivot_idx_int < len(_state(state, 'Node_ID')) else "Unknown"
        print(
            f"  Sensor {i + 1}: Index={pivot_idx}, Node ID={node_id_formatted} "
            f"(raw: {node_id_raw}), Residual={res:.6f}, Relative residual={rel_res:.6f}"
        )
    print("=" * 50 + "\n")
    return True


def handle_option_7(state):
    plot_cumulative_sum(_state(state, 'X'), _state(state, 'S'), get_user_input)
    return True


def handle_option_9(state):
    r_values = [1, 6, 10]
    plot_payload = []
    for event_info in _load_option9_event_matrices(_current_dir(state)):
        reconstructed_by_r = {}
        for r in r_values:
            _, all_x = SVDQR_NSE(event_info['event_matrix'], r, _state(state, 'Psi'), _analysis_context(state))
            reconstructed_by_r[r] = all_x
        plot_payload.append({
            'event_file': event_info['event_file'],
            'event_type': event_info['event_type'],
            'observed_matrix': event_info['event_matrix'],
            'reconstructed_by_r': reconstructed_by_r,
            'time_labels': event_info.get('time_labels', []),
        })
    plot_shadowline(plot_payload, _current_dir(state))
    print(f"Shadow line chart saved to: {os.path.join(_current_dir(state), 'Shadowline.png')}")
    return True


def handle_option_10(state):
    r_values = range(1, int(get_user_input("Enter r value, press Enter to skip", 10)) + 1)
    testing_event_matrices = _state(state, 'testing_event_matrices')
    real_event_matrices = [event for event in testing_event_matrices if '200-year' not in event['event_name'].lower()]
    design_event_matrices = [event for event in testing_event_matrices if '200-year' in event['event_name'].lower()]

    nse_values_prediction = {
        r: Calculate_eventwise_nse_for_r(
            real_event_matrices,
            _state(state, 'Node_ID'),
            r,
            _state(state, 'Psi'),
            _analysis_context(state)
        )
        for r in r_values
    }
    nse_values_prediction_200 = {
        r: Calculate_eventwise_nse_for_r(
            design_event_matrices,
            _state(state, 'Node_ID'),
            r,
            _state(state, 'Psi'),
            _analysis_context(state)
        )
        for r in r_values
    }
    plot_boxplot(
        nse_values_prediction,
        nse_values_prediction_200,
        r_values,
        _current_dir(state),
        ylabel='Node-level NSE',
        output_filename='NSE_Boxplot_Per_Node.png'
    )
    csv_output_root, generated_csvs = _export_option10_nodewise_nse_csvs(
        state,
        real_event_matrices,
        design_event_matrices,
        r_values
    )
    print(f"Per-node NSE boxplot saved to: {os.path.join(_current_dir(state), 'NSE_Boxplot_Per_Node.png')}")
    print(f"Node-level NSE CSV files saved to: {csv_output_root}")
    if generated_csvs:
        print("Generated CSV files:")
        for csv_path in generated_csvs[:6]:
            print(f"  {csv_path}")
        if len(generated_csvs) > 6:
            print(f"  ... and {len(generated_csvs) - 6} more files")
    return True


def handle_option_11(state):
    r_values = range(1, int(get_user_input("Enter max r value, press Enter to skip", 10)) + 1)
    psi_basis, _, _ = load_svd_decomposition(
        _current_dir(state),
        _state(state, 'X'),
        _state(state, 'X_mean'),
        _state(state, 'X_std_safe'),
        _state(state, 'normalization_mode')
    )
    reconstruction_context = dict(_analysis_context(state))
    print("The lower hexbin panels use all time points from the six event files.")
    testing_event_matrices = _state(state, 'testing_event_matrices')
    real_event_matrices = [event for event in testing_event_matrices if '200-year' not in event['event_name'].lower()]
    design_event_matrices = [event for event in testing_event_matrices if '200-year' in event['event_name'].lower()]

    nse_values_prediction = {
        r: Calculate_eventwise_overall_nse_for_r(
            real_event_matrices,
            _state(state, 'Node_ID'),
            r,
            psi_basis,
            reconstruction_context
        )
        for r in r_values
    }
    nse_values_prediction_200 = {
        r: Calculate_eventwise_overall_nse_for_r(
            design_event_matrices,
            _state(state, 'Node_ID'),
            r,
            psi_basis,
            reconstruction_context
        )
        for r in r_values
    }

    mode_suffix = _state(state, 'normalization_mode')
    boxplot_output_name = f'System_Level_Reconstruction_NSE_Boxplot_{mode_suffix}.png'
    plot_boxplot(
        nse_values_prediction,
        nse_values_prediction_200,
        r_values,
        _current_dir(state),
        ylabel='System-level NSE',
        output_filename=boxplot_output_name,
        real_color='#4a98c5',
        design_color='#dc6d57'
    )
    output_name = f'System_Level_Reconstruction_NSE_Line_{mode_suffix}.png'
    plot_event_nse_linechart(
        nse_values_prediction,
        nse_values_prediction_200,
        r_values,
        _current_dir(state),
        ylabel='System-level NSE',
        output_filename=output_name
    )
    overall_real_csv = _save_System_Level_Reconstruction_event_nse_tables(
        _current_dir(state),
        f'System_Level_Reconstruction_NSE_Observed_Events_{mode_suffix}.csv',
        real_event_matrices,
        nse_values_prediction
    )
    overall_200_csv = _save_System_Level_Reconstruction_event_nse_tables(
        _current_dir(state),
        f'System_Level_Reconstruction_NSE_200Year_{mode_suffix}.csv',
        design_event_matrices,
        nse_values_prediction_200
    )

    merged_truth = _load_System_Level_Reconstruction_scatter_truth_matrix(_current_dir(state))
    representative_r = _representative_r_values(r_values)
    scatter_datasets = []
    for r in representative_r:
        _, reconstructed_columns = SVDQR_NSE(merged_truth, r, psi_basis, reconstruction_context)
        reconstructed_matrix = np.asarray(reconstructed_columns).T
        pooled_nse = Calculate_eventwise_overall_nse_for_r(
            testing_event_matrices,
            _state(state, 'Node_ID'),
            r,
            psi_basis,
            reconstruction_context
        )
        pooled_nse_mean = float(np.mean(pooled_nse)) if pooled_nse else np.nan
        scatter_datasets.append({
            'label': f'r={r}, mean system-level NSE={pooled_nse_mean:.3f}',
            'true_values': merged_truth,
            'pred_values': reconstructed_matrix,
        })

    combined_figure_path = plot_System_Level_Reconstruction_combined_figure(
        nse_values_prediction,
        nse_values_prediction_200,
        r_values,
        scatter_datasets,
        _current_dir(state),
        output_filename=f'System_Level_Reconstruction_Combined_Line_Hexbin_{mode_suffix}.png'
    )
    combined_boxplot_figure_path = plot_System_Level_Reconstruction_combined_boxplot_scatter(
        nse_values_prediction,
        nse_values_prediction_200,
        r_values,
        scatter_datasets,
        _current_dir(state),
        output_filename=f'System_Level_Reconstruction_Combined_Boxplot_Hexbin_{mode_suffix}.png'
    )

    print(f"System-level NSE boxplot saved to: {os.path.join(_current_dir(state), boxplot_output_name)}")
    print(f"System-level NSE line chart saved to: {os.path.join(_current_dir(state), output_name)}")
    print(f"Combined boxplot-hexbin figure saved to: {combined_boxplot_figure_path}")
    print(f"Combined line-hexbin figure saved to: {combined_figure_path}")
    print(f"Real-event system-level NSE table saved to: {overall_real_csv}")
    print(f"200-year system-level NSE table saved to: {overall_200_csv}")
    return True


def handle_option_12(state):
    print("\n========== Ambient Noise Analysis (Gaussian Noise Model) ==========")
    print("Description:")
    print("  - Noise type: Multiplicative Gaussian Noise")
    print("  - Noise model: y_noisy = y * (1 + N(0, sigma)), sigma = noise_level / 3")
    print("  - 5% noise: 99.7% of noise values within ±5% (3-sigma principle)")
    print("  - Monte Carlo sampling characterizes the statistical properties of noise")
    print("=" * 50 + "\n")

    max_r = int(get_user_input("Enter r value, press Enter to skip", 10))
    r_values = range(1, max_r + 1)
    num_monte_carlo = int(get_user_input("Enter number of Monte Carlo samples (recommended 1000-10000, default 1000)", 1000))
    normalization_mode = _state(state, 'normalization_mode')
    system_level_cache_path = _system_level_noise_cache_path(state, normalization_mode, max_r, num_monte_carlo)
    use_cache, _, _ = _prompt_use_existing_cache(
        state,
        system_level_cache_path,
        "A valid System-level noise cache exists. Use it directly for plotting? (y/n)",
        default='y',
        missing_message=f"Existing System-level noise cache is incomplete and will not be used: {system_level_cache_path}",
    )

    if use_cache:
        print(f"\nLoading System-level noise cache: {system_level_cache_path}")
        cache_payload = _load_system_level_noise_cache(system_level_cache_path)
        output_path = plot_noisy_system_boxplot(
            cache_payload['clean'],
            cache_payload['5%'],
            cache_payload['10%'],
            cache_payload['15%'],
            r_values,
            _current_dir(state),
            output_filename='Reconstructed_x_vs_y_Noisy_System_Level_NSE.png'
        )
        summary_path = _save_system_level_noise_summary(
            _current_dir(state),
            normalization_mode,
            num_monte_carlo,
            cache_payload
        )
        print(f"System-level NSE figure saved to: {output_path}")
        print(f"System-level noise summary saved to: {summary_path}")
        return True

    if get_user_input("Recompute system-level noise distributions from cached event matrices? (y/n)", "y").lower() != 'y':
        print("Ambient noise analysis cancelled.")
        return False

    nse_values_prediction = {}
    nse_values_prediction_noisy_5 = {}
    nse_values_prediction_noisy_10 = {}
    nse_values_prediction_noisy_15 = {}
    overall_nse_values_clean = {}
    overall_nse_values_noisy_5 = {}
    overall_nse_values_noisy_10 = {}
    overall_nse_values_noisy_15 = {}
    testing_event_matrices = _state(state, 'testing_event_matrices')
    print(f"\nRunning ambient noise analysis (Monte Carlo sampling: {num_monte_carlo} iterations)...")
    for r in r_values:
        print(f"  Processing r={r}...")
        noise_samples_by_level = {}
        for noise_level in (0.05, 0.10, 0.15):
            per_event_samples = {}
            for event_item in testing_event_matrices:
                event_tag = event_item['event_name'].replace(' ', '_').replace('/', '_')
                per_event_samples[event_item['event_name']] = _load_or_create_noise_samples(
                    state,
                    event_tag,
                    event_item['data'],
                    r,
                    noise_level,
                    num_monte_carlo
                )
            noise_samples_by_level[noise_level] = per_event_samples

        noisy_result_5 = Calculate_eventwise_nse_noisy(
            testing_event_matrices,
            _state(state, 'Node_ID'),
            r,
            0.05,
            _state(state, 'Psi'),
            _analysis_context(state),
            num_monte_carlo=num_monte_carlo,
            noise_samples_by_event=noise_samples_by_level[0.05]
        )
        noisy_result_10 = Calculate_eventwise_nse_noisy(
            testing_event_matrices,
            _state(state, 'Node_ID'),
            r,
            0.10,
            _state(state, 'Psi'),
            _analysis_context(state),
            num_monte_carlo=num_monte_carlo,
            noise_samples_by_event=noise_samples_by_level[0.10]
        )
        noisy_result_15 = Calculate_eventwise_nse_noisy(
            testing_event_matrices,
            _state(state, 'Node_ID'),
            r,
            0.15,
            _state(state, 'Psi'),
            _analysis_context(state),
            num_monte_carlo=num_monte_carlo,
            noise_samples_by_event=noise_samples_by_level[0.15]
        )
        noisy_overall_5 = Calculate_eventwise_overall_nse_noisy(
            testing_event_matrices,
            _state(state, 'Node_ID'),
            r,
            0.05,
            _state(state, 'Psi'),
            _analysis_context(state),
            num_monte_carlo=num_monte_carlo,
            noise_samples_by_event=noise_samples_by_level[0.05]
        )
        noisy_overall_10 = Calculate_eventwise_overall_nse_noisy(
            testing_event_matrices,
            _state(state, 'Node_ID'),
            r,
            0.10,
            _state(state, 'Psi'),
            _analysis_context(state),
            num_monte_carlo=num_monte_carlo,
            noise_samples_by_event=noise_samples_by_level[0.10]
        )
        noisy_overall_15 = Calculate_eventwise_overall_nse_noisy(
            testing_event_matrices,
            _state(state, 'Node_ID'),
            r,
            0.15,
            _state(state, 'Psi'),
            _analysis_context(state),
            num_monte_carlo=num_monte_carlo,
            noise_samples_by_event=noise_samples_by_level[0.15]
        )
        nse_values_prediction[r] = list(noisy_result_5['nse_clean'])
        nse_values_prediction_noisy_5[r] = [nse for iteration in noisy_result_5['nse_noisy'] for nse in iteration]
        nse_values_prediction_noisy_10[r] = [nse for iteration in noisy_result_10['nse_noisy'] for nse in iteration]
        nse_values_prediction_noisy_15[r] = [nse for iteration in noisy_result_15['nse_noisy'] for nse in iteration]
        overall_nse_values_clean[r] = list(noisy_overall_5['nse_clean'])
        overall_nse_values_noisy_5[r] = [nse for iteration in noisy_overall_5['nse_noisy'] for nse in iteration]
        overall_nse_values_noisy_10[r] = [nse for iteration in noisy_overall_10['nse_noisy'] for nse in iteration]
        overall_nse_values_noisy_15[r] = [nse for iteration in noisy_overall_15['nse_noisy'] for nse in iteration]

    _save_system_level_noise_cache(
        system_level_cache_path,
        overall_nse_values_clean,
        overall_nse_values_noisy_5,
        overall_nse_values_noisy_10,
        overall_nse_values_noisy_15,
    )
    print(f"Saved refreshed System-level noise cache to: {system_level_cache_path}")

    print("Ambient noise analysis complete. Generating System-level NSE figure...")
    output_path = plot_noisy_system_boxplot(
        overall_nse_values_clean,
        overall_nse_values_noisy_5,
        overall_nse_values_noisy_10,
        overall_nse_values_noisy_15,
        r_values,
        _current_dir(state),
        output_filename='Reconstructed_x_vs_y_Noisy_System_Level_NSE.png'
    )
    summary_path = _save_noise_boxplot_summary(
        _current_dir(state),
        _state(state, 'normalization_mode'),
        num_monte_carlo,
        nse_values_prediction,
        nse_values_prediction_noisy_5,
        nse_values_prediction_noisy_10,
        nse_values_prediction_noisy_15,
        overall_nse_values_clean,
        overall_nse_values_noisy_5,
        overall_nse_values_noisy_10,
        overall_nse_values_noisy_15,
    )
    print(f"System-level NSE figure saved to: {output_path}")
    print(f"Noise boxplot summary saved to: {summary_path}")
    return True


def handle_option_13(state):
    r = int(get_user_input("Enter r value, press Enter to skip", 10))
    nse_prediction_d, event_nse_prediction_d, Node_ID_D = SVDQR_NSE_SD(
        r,
        _state(state, 'Psi'),
        _analysis_context(state),
        event_matrices=_state(state, 'testing_event_matrices')
    )
    plot_boxplot_SD(
        nse_prediction_d,
        event_nse_prediction_d,
        r,
        Node_ID_D,
        _current_dir(state),
        ylabel='NSE',
        output_filename='Reconstructed x vs y SD NSE.png'
    )
    print(f"NSE boxplot saved to: {os.path.join(_current_dir(state), 'Reconstructed x vs y SD NSE.png')}")
    return True


def handle_option_14(state):
    r_input = get_user_input("Enter r value(s) for the System-level dropout boxplot, comma-separated", "6,10")
    r_values = [int(item.strip()) for item in str(r_input).split(',') if item.strip()]
    if not r_values:
        print("No valid r values provided.")
        return False

    bundle = _generate_dropout_analysis_bundle(state, r_values, ensure_diagnostics=True)
    _print_dropout_analysis_bundle("Sensor dropout analysis figures generated:", bundle)
    return True


def handle_option_15(state):
    max_r = int(get_user_input("Enter max r value for random benchmark, press Enter to skip", 10))
    exhaustive_r_values = [r for r in range(1, min(4, max_r) + 1)]
    monte_carlo_r_values = [r for r in range(5, max_r + 1)]
    default_workers = _default_exhaustive_workers()
    num_workers = int(get_user_input("Enter number of worker processes for exhaustive r=1-4, press Enter to skip", default_workers))
    cache_name = f"random_benchmark_eventwise_{_state(state, 'normalization_mode')}.npz"
    random_analysis_save_path = _cached_npz_path(state, cache_name)
    should_recompute, cache_exists, cache_missing = _prompt_recompute_cached_results(
        state,
        random_analysis_save_path,
        "Recompute random benchmark results? (y/n)",
        default='n',
        missing_message=f"Existing random benchmark cache is incomplete and will not be used: {random_analysis_save_path}",
    )

    if cache_exists and not cache_missing and not should_recompute:
        saved_data = np.load(random_analysis_save_path, allow_pickle=True)
        dss_eventwise = saved_data['dss_eventwise'].item()
        exhaustive_distribution = saved_data['exhaustive_distribution'].item()
        exhaustive_optimum = saved_data['exhaustive_optimum'].item()
        exhaustive_best_sensors = saved_data['exhaustive_best_sensors'].item()
        monte_carlo_distribution = saved_data['monte_carlo_distribution'].item()
        monte_carlo_iterations = saved_data['monte_carlo_iterations'].item()
        print("Loaded random benchmark results from file.")
    else:
        print("Computing system-level benchmark results for the paper figure layout...")
        testing_event_matrices = _state(state, 'testing_event_matrices')
        dss_eventwise = {
            r: Calculate_eventwise_overall_nse_for_r(
                testing_event_matrices,
                _state(state, 'Node_ID'),
                r,
                _state(state, 'Psi'),
                _analysis_context(state)
            )
            for r in range(1, max_r + 1)
        }
        exhaustive_distribution = {}
        exhaustive_optimum = {}
        exhaustive_best_sensors = {}
        for r in exhaustive_r_values:
            pooled_values, optimum_values, best_sensors, _, _ = SVDQR_NSE_Exhaustive_Eventwise_Overall(
                testing_event_matrices,
                _state(state, 'Node_ID'),
                r,
                _state(state, 'Psi'),
                _analysis_context(state),
                max_combos=2000000,
                num_workers=num_workers,
            )
            exhaustive_distribution[r] = pooled_values or []
            exhaustive_optimum[r] = optimum_values or []
            exhaustive_best_sensors[r] = best_sensors or []

        monte_carlo_distribution = {}
        monte_carlo_iterations = {}
        for r in monte_carlo_r_values:
            iterations = _default_random_mc_iterations(state, r)
            monte_carlo_iterations[r] = iterations
            monte_carlo_distribution[r] = SVDQR_NSE_Random_Eventwise_Overall(
                testing_event_matrices,
                _state(state, 'Node_ID'),
                r,
                _state(state, 'Psi'),
                _analysis_context(state),
                num_iterations=iterations,
            )

        np.savez(
            random_analysis_save_path,
            dss_eventwise=dss_eventwise,
            exhaustive_distribution=exhaustive_distribution,
            exhaustive_optimum=exhaustive_optimum,
            exhaustive_best_sensors=exhaustive_best_sensors,
            monte_carlo_distribution=monte_carlo_distribution,
            monte_carlo_iterations=monte_carlo_iterations,
        )
        print("Random benchmark results saved to file.")

    if exhaustive_r_values:
        plot_exhaustive_eventwise_benchmark(
            {r: exhaustive_distribution[r] for r in exhaustive_r_values},
            {r: dss_eventwise[r] for r in exhaustive_r_values},
            {r: exhaustive_optimum[r] for r in exhaustive_r_values},
            _current_dir(state),
            output_filename=f'Benchmark_Exhaustive_vs_DSS_{_state(state, "normalization_mode")}.png',
            title_suffix=_state(state, 'normalization_label'),
            axis_segments=[(0.80, 1.00), (-1.2, 0.80), (-5.0, -1.2)],
        )
        print(f"Exhaustive benchmark figure saved to: {os.path.join(_current_dir(state), f'Benchmark_Exhaustive_vs_DSS_{_state(state, 'normalization_mode')}.png')}")

    if monte_carlo_r_values:
        pooled_mc_distribution = {
            r: [value for iteration in monte_carlo_distribution[r] for value in iteration]
            for r in monte_carlo_r_values
        }
        plot_random_monte_carlo_eventwise_benchmark(
            pooled_mc_distribution,
            {r: dss_eventwise[r] for r in monte_carlo_r_values},
            _current_dir(state),
            output_filename=f'Benchmark_Random_MC_vs_DSS_{_state(state, "normalization_mode")}.png',
            title_suffix=_state(state, 'normalization_label'),
        )
        print(f"Monte Carlo random benchmark figure saved to: {os.path.join(_current_dir(state), f'Benchmark_Random_MC_vs_DSS_{_state(state, 'normalization_mode')}.png')}")

    if exhaustive_r_values and monte_carlo_r_values:
        _, _, composite_path = create_benchmark_paper_composite(
            normalization_mode=_state(state, 'normalization_mode'),
            current_dir=_current_dir(state),
            model_dir=_model_directory(state),
        )
        print(f"Benchmark paper composite figure saved to: {composite_path}")

    summary_rows = []
    for r in exhaustive_r_values:
        summary_rows.append({
            'r': r,
            'benchmark_type': 'exhaustive',
            'dss_mean_system_level_nse': float(np.mean(dss_eventwise[r])) if dss_eventwise[r] else np.nan,
            'optimum_mean_event_nse': float(np.mean(exhaustive_optimum[r])) if exhaustive_optimum[r] else np.nan,
            'best_sensors': ','.join(map(str, exhaustive_best_sensors.get(r, []))),
            'num_samples': math.comb(len(_state(state, 'Node_ID')), r),
        })
    for r in monte_carlo_r_values:
        flat_values = [value for iteration in monte_carlo_distribution[r] for value in iteration]
        summary_rows.append({
            'r': r,
            'benchmark_type': 'monte_carlo',
            'dss_mean_system_level_nse': float(np.mean(dss_eventwise[r])) if dss_eventwise[r] else np.nan,
            'optimum_mean_event_nse': np.nan,
            'best_sensors': '',
            'num_samples': monte_carlo_iterations[r],
            'mc_mean_event_nse': float(np.mean(flat_values)) if flat_values else np.nan,
            'mc_median_event_nse': float(np.median(flat_values)) if flat_values else np.nan,
        })
    summary_path = os.path.join(_current_dir(state), f'Benchmark_summary_{_state(state, "normalization_mode")}.csv')
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    print(f"Benchmark summary saved to: {summary_path}")
    return True


def handle_option_16(state):
    output_filename = f"Psi_heatmap_{_state(state, 'normalization_mode')}.png"
    plot_psi_heatmap(
        _state(state, 'Psi'),
        _current_dir(state),
        title_suffix=_state(state, 'normalization_label'),
        output_filename=output_filename
    )
    print(f"Psi heatmap saved to: {os.path.join(_current_dir(state), output_filename)}")
    return True


def handle_option_17(state):
    normalization_mode = _state(state, 'normalization_mode')
    dropout_output_dir = os.path.join(_current_dir(state), f'Dropout_Discussion_{normalization_mode}')
    coefficient_path = os.path.join(dropout_output_dir, 'dropout_multifactor_standardized_coefficients.csv')
    if not os.path.exists(coefficient_path):
        print(
            "Function 16 requires dropout diagnostics generated by Function 13. "
            "Please run 13. Sensor dropout analysis first."
        )
        return True

    def _pivot_to_indices(pivots):
        indices = []
        for p in pivots:
            if hasattr(p, 'item'):
                indices.append(int(p.item()))
            elif hasattr(p, '__len__') and not isinstance(p, str):
                indices.append(int(p[0]))
            else:
                indices.append(int(p))
        return indices

    def _post_dropout_condition_numbers(psi, selected_sensors, r):
        values = []
        psi_r = psi[:, :r]
        for drop_index in selected_sensors:
            remaining_sensors = [idx for idx in selected_sensors if idx != drop_index]
            if not remaining_sensors:
                values.append(np.nan)
                continue
            psi_red = psi_r[remaining_sensors, :]
            singular_values = np.linalg.svd(psi_red, compute_uv=False)
            sigma_min = float(np.min(singular_values)) if singular_values.size > 0 else 0.0
            sigma_max = float(np.max(singular_values)) if singular_values.size > 0 else 0.0
            values.append(np.nan if sigma_min == 0.0 else float(sigma_max / sigma_min))
        return np.array(values, dtype=float)

    def _max_modal_loadings(psi, selected_sensors, r):
        psi_r = psi[:, :r]
        return np.array([
            float(np.max(np.abs(psi_r[idx, :]))) for idx in selected_sensors
        ], dtype=float)

    def _within_r_zscore(values):
        values = np.asarray(values, dtype=float)
        finite_values = values[np.isfinite(values)]
        if finite_values.size == 0:
            return np.full(values.shape, np.nan, dtype=float)
        std = float(np.nanstd(values, ddof=0))
        if not np.isfinite(std) or std <= 0.0:
            return np.zeros(values.shape, dtype=float)
        return (values - float(np.nanmean(values))) / std

    def _load_dropout_standardized_coefficients(normalization_mode):
        output_dir = os.path.join(_current_dir(state), f'Dropout_Discussion_{normalization_mode}')
        coefficient_path = os.path.join(output_dir, 'dropout_multifactor_standardized_coefficients.csv')
        if not os.path.exists(coefficient_path):
            raise FileNotFoundError(
                "Function 16 requires dropout diagnostics generated by Function 13. "
                "Please run 13. Sensor dropout analysis first."
            )

        coefficient_df = pd.read_csv(coefficient_path)
        coefficients = dict(zip(coefficient_df['feature'], coefficient_df['coefficient']))
        required_features = [
            'Relative_Projection_Residual_z',
            'Condition_Number_z',
            'max_abs_loading_z',
            'pivot_order_z',
        ]
        missing_features = [feature for feature in required_features if feature not in coefficients]
        if missing_features:
            raise ValueError(
                'Missing standardized dropout coefficients: ' + ', '.join(missing_features)
            )
        return coefficients, coefficient_path

    results = {}
    psi = _state(state, 'Psi')
    for r in range(2, 11):
        Pivot, residuals, relative_residuals = SVDQR(_state(state, 'X'), psi, _state(state, 'S'), r)
        selected_sensors = _pivot_to_indices(Pivot[:len(relative_residuals)])
        results[r] = {
            'Pivot': Pivot,
            'residuals': residuals,
            'relative_residuals': relative_residuals,
            'condition_numbers': _post_dropout_condition_numbers(psi, selected_sensors, r),
            'max_modal_loadings': _max_modal_loadings(psi, selected_sensors, r),
        }
    all_relative_residuals = [results[r]['relative_residuals'] for r in range(2, 11)]
    all_condition_numbers = [results[r]['condition_numbers'] for r in range(2, 11)]
    all_max_modal_loadings = [results[r]['max_modal_loadings'] for r in range(2, 11)]
    all_pivots = [results[r]['Pivot'] for r in range(2, 11)]

    heatmap_specs = [
        (
            all_relative_residuals,
            f"residuals_heatmap_{normalization_mode}.png",
            'Relative Projection Residual',
            'Residuals heatmap',
        ),
        (
            all_condition_numbers,
            f"condition_number_heatmap_{normalization_mode}.png",
            'Post-dropout condition number (K)',
            'Condition-number heatmap',
        ),
        (
            all_max_modal_loadings,
            f"max_modal_loading_heatmap_{normalization_mode}.png",
            'Maximum modal loading (L)',
            'Maximum-modal-loading heatmap',
        ),
    ]

    font_settings = None
    for metric_values, output_filename, colorbar_label, print_label in heatmap_specs:
        font_settings = plot_residuals_heatmap(
            _state(state, 'Node_ID'),
            all_pivots,
            metric_values,
            _current_dir(state),
            title_suffix=_state(state, 'normalization_label'),
            output_filename=output_filename,
            colorbar_label=colorbar_label,
            font_settings=font_settings
        )
        print(f"{print_label} saved to: {os.path.join(_current_dir(state), output_filename)}")

    coefficients, coefficient_path = _load_dropout_standardized_coefficients(normalization_mode)
    print(f"Using standardized dropout coefficients from: {coefficient_path}")

    contribution_specs = [
        (
            [
                coefficients['Relative_Projection_Residual_z'] * _within_r_zscore(results[r]['relative_residuals'])
                for r in range(2, 11)
            ],
            f"rpr_model_contribution_heatmap_{normalization_mode}.png",
            r'Model contribution ($\beta_{\mathrm{RPR}} z_{\mathrm{RPR}}$)',
            'RPR model-contribution heatmap',
        ),
        (
            [
                coefficients['Condition_Number_z'] * _within_r_zscore(results[r]['condition_numbers'])
                for r in range(2, 11)
            ],
            f"condition_number_model_contribution_heatmap_{normalization_mode}.png",
            r'Model contribution ($\beta_{K} z_{K}$)',
            'Condition-number model-contribution heatmap',
        ),
        (
            [
                coefficients['max_abs_loading_z'] * _within_r_zscore(results[r]['max_modal_loadings'])
                for r in range(2, 11)
            ],
            f"max_modal_loading_model_contribution_heatmap_{normalization_mode}.png",
            r'Model contribution ($\beta_{L} z_{L}$)',
            'Maximum-modal-loading model-contribution heatmap',
        ),
        (
            [
                coefficients['pivot_order_z'] * _within_r_zscore(np.arange(1, len(results[r]['relative_residuals']) + 1))
                for r in range(2, 11)
            ],
            f"qr_rank_model_contribution_heatmap_{normalization_mode}.png",
            r'Model contribution ($\beta_{\mathrm{QR}} z_{\mathrm{QR}}$)',
            'QR-rank model-contribution heatmap',
        ),
    ]
    contribution_values = [
        values
        for metric_values, _, _, _ in contribution_specs
        for values in metric_values
    ]
    finite_contributions = np.concatenate([
        np.asarray(values, dtype=float)[np.isfinite(values)]
        for values in contribution_values
        if np.asarray(values, dtype=float)[np.isfinite(values)].size > 0
    ])
    contribution_limit = (
        float(np.nanmax(np.abs(finite_contributions)))
        if finite_contributions.size > 0
        else None
    )

    for metric_values, output_filename, colorbar_label, print_label in contribution_specs:
        plot_residuals_heatmap(
            _state(state, 'Node_ID'),
            all_pivots,
            metric_values,
            _current_dir(state),
            title_suffix=_state(state, 'normalization_label'),
            output_filename=output_filename,
            colorbar_label=colorbar_label,
            font_settings=font_settings,
            cmap='RdBu_r',
            vmin=-contribution_limit if contribution_limit is not None else None,
            vmax=contribution_limit if contribution_limit is not None else None,
            center=0.0,
        )
        print(f"{print_label} saved to: {os.path.join(_current_dir(state), output_filename)}")

    return True


def handle_option_18(state):
    dataset_scope = get_user_input(
        "Select data source for hydrograph plotting (training/testing, default: testing)",
        'testing'
    ).strip().lower()
    name = get_user_input(
        "Enter event name or prefix, press Enter to skip (e.g., 8-26, 9-3, 5-21, 6-15, 9-5, 200-year)",
        '9-5'
    )
    selected_nodes = get_user_input(
        "Enter node IDs to plot (comma-separated), press Enter to skip",
        ','.join(_swmm_config(state)['TARGET_NODE_IDS'])
    ).split(',')
    flow_data_path, rainfall_data_path = _resolve_hydrograph_paths(_current_dir(state), dataset_scope, name)
    if not flow_data_path:
        raise FileNotFoundError(f"No matching flowrate file found for '{name}' in the selected dataset.")
    if not rainfall_data_path:
        raise FileNotFoundError(f"No matching rainfall file found for '{name}' in the selected dataset.")
    print(f"Using flow data file: {os.path.basename(flow_data_path)}")
    print(f"Using rainfall data file: {os.path.basename(rainfall_data_path)}")

    flow_ylim = _parse_axis_range(get_user_input(
        "Enter flow y-axis range as min,max (press Enter to use default)",
        ''
    ))
    rainfall_ylim = _parse_axis_range(get_user_input(
        "Enter rainfall y-axis range as min,max (press Enter to use default)",
        ''
    ))
    event_tag = os.path.splitext(os.path.basename(flow_data_path))[0].replace('Flowrates_', '')
    output_filename = f"rainfall_flowrate_{event_tag}.png"

    plot_hydrograph_with_rainfall(
        flow_csv_path=flow_data_path,
        rainfall_excel_path=rainfall_data_path,
        current_dir=_current_dir(state),
        selected_nodes=selected_nodes,
        rainfall_column=0,
        flow_ylim=flow_ylim,
        rainfall_ylim=rainfall_ylim,
        output_filename=output_filename,
    )
    return True


def handle_option_19(state):
    r = int(get_user_input("Enter a single r value for exhaustive search, press Enter to skip", 4))
    default_workers = _default_exhaustive_workers()
    num_workers = int(get_user_input("Enter number of worker processes for exhaustive search, press Enter to skip", default_workers))
    testing_event_matrices = _state(state, 'testing_event_matrices')
    single_r_cache_path = _single_r_exhaustive_cache_path(state, r)
    benchmark_cache_path = _cached_npz_path(state, f"random_benchmark_eventwise_{_state(state, 'normalization_mode')}.npz")
    dss_eventwise = {}
    exhaustive_distribution = {}
    exhaustive_optimum = {}
    exhaustive_best_sensors = {}
    evaluated_combos = math.comb(len(_state(state, 'Node_ID')), r)
    elapsed_seconds = np.nan

    if os.path.exists(single_r_cache_path) and not _analysis_cache_missing(state, single_r_cache_path):
        single_r_cache = np.load(single_r_cache_path, allow_pickle=True)
        dss_eventwise[r] = single_r_cache['dss_eventwise'].tolist()
        exhaustive_distribution[r] = single_r_cache['exhaustive_distribution'].tolist()
        exhaustive_optimum[r] = single_r_cache['exhaustive_optimum'].tolist()
        exhaustive_best_sensors[r] = single_r_cache['exhaustive_best_sensors'].tolist()
        if 'evaluated_combinations' in single_r_cache:
            evaluated_combos = int(single_r_cache['evaluated_combinations'])
        print(f"Using single-r exhaustive cache from: {single_r_cache_path}")
    elif os.path.exists(benchmark_cache_path) and not _analysis_cache_missing(state, benchmark_cache_path):
        cached_benchmark = np.load(benchmark_cache_path, allow_pickle=True)
        cached_dss_eventwise = cached_benchmark['dss_eventwise'].item()
        cached_exhaustive_distribution = cached_benchmark['exhaustive_distribution'].item()
        cached_exhaustive_optimum = cached_benchmark['exhaustive_optimum'].item()
        cached_exhaustive_best_sensors = cached_benchmark['exhaustive_best_sensors'].item()
        if r in cached_dss_eventwise and r in cached_exhaustive_distribution and r in cached_exhaustive_optimum:
            dss_eventwise[r] = cached_dss_eventwise[r]
            exhaustive_distribution[r] = cached_exhaustive_distribution[r]
            exhaustive_optimum[r] = cached_exhaustive_optimum[r]
            exhaustive_best_sensors[r] = cached_exhaustive_best_sensors.get(r, [])
            print(f"Using benchmark cache entry for r={r} from: {benchmark_cache_path}")

    if r not in dss_eventwise:
        dss_eventwise[r] = Calculate_eventwise_overall_nse_for_r(
            testing_event_matrices,
            _state(state, 'Node_ID'),
            r,
            _state(state, 'Psi'),
            _analysis_context(state)
        )
        pooled_values, optimum_values, best_sensors, evaluated_combos, elapsed_seconds = (
            SVDQR_NSE_Exhaustive_Eventwise_Overall(
                testing_event_matrices,
                _state(state, 'Node_ID'),
                r,
                _state(state, 'Psi'),
                _analysis_context(state),
                max_combos=2000000,
                num_workers=num_workers,
            )
        )
        if pooled_values is None:
            print(f"Exhaustive benchmark was skipped for r={r} because the combination count exceeded the configured limit.")
            return True
        exhaustive_distribution[r] = pooled_values
        exhaustive_optimum[r] = optimum_values
        exhaustive_best_sensors[r] = best_sensors
        np.savez(
            single_r_cache_path,
            dss_eventwise=np.asarray(dss_eventwise[r], dtype=float),
            exhaustive_distribution=np.asarray(exhaustive_distribution[r], dtype=float),
            exhaustive_optimum=np.asarray(exhaustive_optimum[r], dtype=float),
            exhaustive_best_sensors=np.asarray(exhaustive_best_sensors[r], dtype=object),
            evaluated_combinations=np.asarray(evaluated_combos, dtype=int),
        )
        print(f"Single-r exhaustive cache saved to: {single_r_cache_path}")

    print(f"\n=== Exhaustive Benchmark Results (r={r}) ===")
    print(f"Worker processes used: {num_workers}")
    print(f"DSS mean system-level NSE: {float(np.mean(dss_eventwise[r])):.6f}")
    print(f"Exhaustive optimum mean system-level NSE: {float(np.mean(exhaustive_optimum[r])):.6f}")
    print(f"Exhaustive optimum sensor locations: {exhaustive_best_sensors[r]}")
    print(f"Valid combinations evaluated: {evaluated_combos}")
    print(f"Exhaustive search elapsed time: {elapsed_seconds:.2f}s")
    print(f"Mean system-level NSE gap: {abs(float(np.mean(dss_eventwise[r])) - float(np.mean(exhaustive_optimum[r]))):.6f}")

    output_filename = f'Exhaustive_search_r{r}_{_state(state, "normalization_mode")}.png'
    plot_exhaustive_eventwise_gap_only(
        dss_eventwise,
        exhaustive_optimum,
        _current_dir(state),
        output_filename=output_filename,
        title_suffix=_state(state, 'normalization_label'),
    )
    figure_path = os.path.join(_current_dir(state), output_filename)
    print(f"Exhaustive benchmark figure saved to: {figure_path}")

    summary_path = os.path.join(_current_dir(state), f'Exhaustive_search_r{r}_{_state(state, "normalization_mode")}.csv')
    pd.DataFrame([{
        'r': r,
        'dss_mean_system_level_nse': float(np.mean(dss_eventwise[r])),
        'optimum_mean_event_nse': float(np.mean(exhaustive_optimum[r])),
        'system_level_gap_median': float(np.median(np.asarray(exhaustive_optimum[r]) - np.asarray(dss_eventwise[r]))),
        'best_sensors': ','.join(map(str, exhaustive_best_sensors[r])),
        'evaluated_combinations': evaluated_combos,
        'elapsed_seconds': elapsed_seconds,
    }]).to_csv(summary_path, index=False)
    print(f"Exhaustive benchmark summary saved to: {summary_path}")
    return True


def handle_option_20(state):
    r_input = get_user_input(
        "Enter r value(s) for per-event reconstruction diagnostics, comma-separated",
        '6,10'
    )
    r_values = [int(item.strip()) for item in str(r_input).split(',') if item.strip()]
    output_root_name = f"Event_Reconstruction_{_state(state, 'normalization_mode')}"
    run_event_reconstruction_analysis(
        r_values=r_values,
        current_dir=_current_dir(state),
        X=_state(state, 'X'),
        X_mean=_state(state, 'X_mean'),
        X_std_safe=_state(state, 'X_std_safe'),
        X_max_limits=_state(state, 'X_max_limits'),
        node_ids=_state(state, 'Node_ID'),
        nse_func=NSE,
        normalization_mode=_state(state, 'normalization_mode'),
        output_root_name=output_root_name,
        log_prefix='Event reconstruction'
    )
    print(f"Per-event reconstruction diagnostics saved under: {os.path.join(_current_dir(state), output_root_name)}")
    return True


def handle_option_21(state):
    max_r = int(get_user_input("Enter max r value, press Enter to skip", 10))
    run_sensor_stability_analysis(
        _state(state, 'Psi'),
        _state(state, 'S'),
        _current_dir(state),
        max_r=max_r,
        mode_label='Default SVD Basis'
    )
    return True


def handle_option_22(state):
    testing_event_matrices = _state(state, 'testing_event_matrices')
    real_event_matrices = [event for event in testing_event_matrices if '200-year' not in event['event_name'].lower()]
    design_event_matrices = [event for event in testing_event_matrices if '200-year' in event['event_name'].lower()]
    x_matrix = _state(state, 'X')
    node_ids = _state(state, 'Node_ID')
    rows = []

    per_mode = {}
    for mode in ['global_minmax', 'zscore']:
        context = _mode_specific_context(state, mode)
        psi_basis, _, _ = load_svd_decomposition(
            _current_dir(state),
            x_matrix,
            context['X_mean'],
            context['X_std_safe'],
            mode
        )
        per_mode[mode] = {
            'context': context,
            'psi': psi_basis,
        }

    for r in range(1, 11):
        gm_context = per_mode['global_minmax']['context']
        gm_psi = per_mode['global_minmax']['psi']
        zs_context = per_mode['zscore']['context']
        zs_psi = per_mode['zscore']['psi']

        gm_obs_system = Calculate_eventwise_overall_nse_for_r(real_event_matrices, node_ids, r, gm_psi, gm_context)
        zs_obs_system = Calculate_eventwise_overall_nse_for_r(real_event_matrices, node_ids, r, zs_psi, zs_context)
        gm_200_system = Calculate_eventwise_overall_nse_for_r(design_event_matrices, node_ids, r, gm_psi, gm_context)
        zs_200_system = Calculate_eventwise_overall_nse_for_r(design_event_matrices, node_ids, r, zs_psi, zs_context)

        gm_obs_node = np.asarray(Calculate_eventwise_nse_for_r(real_event_matrices, node_ids, r, gm_psi, gm_context), dtype=float)
        zs_obs_node = np.asarray(Calculate_eventwise_nse_for_r(real_event_matrices, node_ids, r, zs_psi, zs_context), dtype=float)
        gm_200_node = np.asarray(Calculate_eventwise_nse_for_r(design_event_matrices, node_ids, r, gm_psi, gm_context), dtype=float)
        zs_200_node = np.asarray(Calculate_eventwise_nse_for_r(design_event_matrices, node_ids, r, zs_psi, zs_context), dtype=float)

        gm_sel = [node_ids[i] for i in _selected_sensors_for_r(gm_psi, r)]
        zs_sel = [node_ids[i] for i in _selected_sensors_for_r(zs_psi, r)]
        overlap_nodes = sorted(set(gm_sel) & set(zs_sel))

        rows.append({
            'r': r,
            'gm_obs_mean': float(np.mean(gm_obs_system)),
            'gm_obs_std': float(np.std(gm_obs_system)),
            'zs_obs_mean': float(np.mean(zs_obs_system)),
            'zs_obs_std': float(np.std(zs_obs_system)),
            'gm_200year': float(np.mean(gm_200_system)),
            'zs_200year': float(np.mean(zs_200_system)),
            'gm_node_obs_median': float(np.median(gm_obs_node)),
            'gm_node_obs_q1': float(np.percentile(gm_obs_node, 25)),
            'gm_node_obs_q3': float(np.percentile(gm_obs_node, 75)),
            'zs_node_obs_median': float(np.median(zs_obs_node)),
            'zs_node_obs_q1': float(np.percentile(zs_obs_node, 25)),
            'zs_node_obs_q3': float(np.percentile(zs_obs_node, 75)),
            'gm_node_200year_median': float(np.median(gm_200_node)),
            'zs_node_200year_median': float(np.median(zs_200_node)),
            'overlap_count': len(overlap_nodes),
            'overlap_fraction': len(overlap_nodes) / float(r),
            'gm_selection': ', '.join(gm_sel),
            'zs_selection': ', '.join(zs_sel),
            'overlap_nodes': ', '.join(overlap_nodes),
        })

    summary_df = pd.DataFrame(rows)
    summary_csv = os.path.join(_current_dir(state), 'Normalization_Comparison_Summary.csv')
    summary_df.to_csv(summary_csv, index=False)
    figure_path = plot_normalization_performance_comparison(
        summary_df,
        _current_dir(state),
        output_filename='Normalization_Comparison_Performance.png'
    )
    print(f"Normalization comparison figure saved to: {figure_path}")
    print(f"Normalization comparison summary saved to: {summary_csv}")
    return True


def handle_option_23(state):
    normalization_mode = _state(state, 'normalization_mode')
    sample_sizes = [int(item.strip()) for item in str(get_user_input(
        "Enter Monte Carlo sample sizes for convergence analysis, comma-separated",
        '1000,5000,10000,20000'
    )).split(',') if item.strip()]
    seeds = [int(item.strip()) for item in str(get_user_input(
        "Enter random seeds, comma-separated",
        '11,22,33'
    )).split(',') if item.strip()]
    r_values = [int(item.strip()) for item in str(get_user_input(
        "Enter r values for convergence analysis, comma-separated",
        '1,2,3,4,5,6,7,8,9,10'
    )).split(',') if item.strip()]
    exact_max_r = int(get_user_input("Enter max r for exact enumeration, press Enter to skip", 4))
    include_design_event = get_user_input(
        "Include 200-year design event in convergence analysis? (y/n, default: y)",
        'y'
    ).strip().lower() != 'n'
    tol_main = float(get_user_input("Enter main-quantile tolerance, press Enter to skip", 0.01))
    tol_tail = float(get_user_input("Enter tail-quantile tolerance, press Enter to skip", 0.02))

    output_dir = os.path.join(_current_dir(state), 'Random_Sampling_Convergence')
    os.makedirs(output_dir, exist_ok=True)

    testing_event_matrices = _state(state, 'testing_event_matrices')
    events = testing_event_matrices if include_design_event else [
        event for event in testing_event_matrices if '200-year' not in event['event_name'].lower()
    ]
    context = _analysis_context(state)
    psi = _state(state, 'Psi')
    node_ids = _state(state, 'Node_ID')

    detail_rows = []
    summary_rows = []
    reference_rows = []

    for r in r_values:
        combo_count = math.comb(len(node_ids), r)
        print(f'Processing convergence diagnostics for r={r} | combinations={combo_count}')

        dss_sensors = np.sort(_selected_sensors_for_r(psi, r))
        dss_values = _sensor_set_event_nse(dss_sensors, events, psi, r, context)
        dss_stats = _summarize_distribution(dss_values)

        exact_stats = None
        if r <= exact_max_r:
            exact_values = _exact_random_distribution(r, events, psi, context)
            exact_stats = _summarize_distribution(exact_values)
            reference_rows.append({
                'r': r,
                'reference_type': 'exact',
                'sample_size': combo_count,
                **exact_stats,
            })

        per_size_stats = {}
        for sample_size in sorted(sample_sizes):
            seed_stats = {}
            for seed in seeds:
                values = _monte_carlo_random_distribution(r, sample_size, seed, events, psi, context)
                stats = _summarize_distribution(values)
                seed_stats[seed] = stats
                detail_rows.append({
                    'r': r,
                    'sample_size': sample_size,
                    'seed': seed,
                    'combo_count': combo_count,
                    **stats,
                })
            per_size_stats[sample_size] = seed_stats

        if exact_stats is not None:
            reference_stats = {seed: exact_stats for seed in seeds}
        else:
            max_size = max(sample_sizes)
            reference_stats = per_size_stats[max_size]
            reference_rows.append({
                'r': r,
                'reference_type': 'largest_mc',
                'sample_size': max_size,
                **{
                    key: float(np.mean([reference_stats[seed][key] for seed in reference_stats]))
                    for key in ['count', 'mean', 'std', 'p05', 'q1', 'median', 'q3', 'p95']
                },
            })

        recommendation = _recommend_random_sample_size(
            r,
            combo_count,
            per_size_stats,
            reference_stats,
            tol_main,
            tol_tail,
        )

        for sample_size, seed_stats in per_size_stats.items():
            avg_stats = {
                key: float(np.mean([seed_stats[seed][key] for seed in seed_stats]))
                for key in ['count', 'mean', 'std', 'p05', 'q1', 'median', 'q3', 'p95']
            }
            main_range = max(
                max(seed_stats[seed][stat_name] for seed in seed_stats) - min(seed_stats[seed][stat_name] for seed in seed_stats)
                for stat_name in ['q1', 'median', 'q3']
            )
            tail_range = max(
                max(seed_stats[seed][stat_name] for seed in seed_stats) - min(seed_stats[seed][stat_name] for seed in seed_stats)
                for stat_name in ['p05', 'p95']
            )
            if exact_stats is not None:
                reference_delta = _pooled_stat_delta(avg_stats, exact_stats)
            else:
                reference_mean_stats = {
                    key: float(np.mean([reference_stats[seed][key] for seed in reference_stats]))
                    for key in ['p05', 'q1', 'median', 'q3', 'p95']
                }
                reference_delta = _pooled_stat_delta(avg_stats, reference_mean_stats)

            summary_rows.append({
                'r': r,
                'combo_count': combo_count,
                'sample_size': sample_size,
                'reference_type': 'exact' if exact_stats is not None else 'largest_mc',
                'main_range_across_seeds': main_range,
                'tail_range_across_seeds': tail_range,
                'max_abs_delta_to_reference': reference_delta,
                'recommended_for_r': recommendation,
                'dss_mean_system_level_nse': dss_stats['mean'],
                'dss_median_event_nse': dss_stats['median'],
                **avg_stats,
            })

    details_path = os.path.join(output_dir, f'Random_Sampling_Convergence_Details_{normalization_mode}.csv')
    summary_path = os.path.join(output_dir, f'Random_Sampling_Convergence_Summary_{normalization_mode}.csv')
    reference_path = os.path.join(output_dir, f'Random_Sampling_Convergence_Reference_{normalization_mode}.csv')
    detail_df = pd.DataFrame(detail_rows)
    summary_df = pd.DataFrame(summary_rows)
    reference_df = pd.DataFrame(reference_rows)
    detail_df.to_csv(details_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    reference_df.to_csv(reference_path, index=False)
    figure_path = plot_random_sampling_convergence_summary(
        summary_df,
        output_dir,
        output_filename=f'Random_Sampling_Convergence_{normalization_mode}.png'
    )

    print(f"Convergence figure saved to: {figure_path}")
    print(f"Convergence detail table saved to: {details_path}")
    print(f"Convergence summary table saved to: {summary_path}")
    print(f"Convergence reference table saved to: {reference_path}")
    return True


def handle_option_24(state):
    print("Exiting program.")
    return False


def handle_invalid_option(state):
    print("Invalid option. Please try again.")
    return True


HANDLERS = {
    '1': handle_option_1,
    '2': handle_option_2,
    '3': handle_option_3,
    '4': handle_option_4,
    '5': handle_option_5,
    '6': handle_option_6,
    '7': handle_option_7,
    '8': handle_option_9,
    '9': handle_option_10,
    '10': handle_option_11,
    '11': handle_option_12,
    '12': handle_option_13,
    '13': handle_option_14,
    '14': handle_option_15,
    '15': handle_option_16,
    '16': handle_option_17,
    '17': handle_option_18,
    '18': handle_option_19,
    '19': handle_option_20,
    '20': handle_option_21,
    '21': handle_option_22,
    '22': handle_option_23,
    '23': handle_option_24,
}


def dispatch_menu_choice(choice, state):
    handler = HANDLERS.get(choice, handle_invalid_option)
    return handler(state)
