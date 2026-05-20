import math
import multiprocessing as mp
import re
import time
from collections import defaultdict
from itertools import combinations

import numpy as np
from scipy.linalg import norm, qr


def Max_Min(X):
    X_min = np.min(X, axis=0)
    X_max = np.max(X, axis=0)
    X_norm = (X - X_min) / (X_max - X_min)
    return X_norm


def Z_score(X):
    X_mean = np.mean(X, axis=0, keepdims=True)
    X_std = np.std(X, axis=0, keepdims=True)
    X_std[X_std == 0] = 1
    X_norm = (X - X_mean) / X_std
    return X_norm


def NSE(x, y):
    x = np.asarray(x).flatten()
    y = np.asarray(y).flatten()
    if x.shape != y.shape:
        raise ValueError(f"Shape mismatch: x.shape={x.shape}, y.shape={y.shape}")
    if np.any(~np.isfinite(x)) or np.any(~np.isfinite(y)):
        raise ValueError("Input contains NaN or Inf values")

    numerator = np.sum((y - x) ** 2)
    denominator = np.sum((y - np.mean(y)) ** 2)
    if denominator == 0:
        return 1.0 if numerator == 0 else 0.0
    return 1.0 - (numerator / denominator)


def _ctx(context, key):
    return context[key]


def _overflow_nodes():
    return {'J122', 'J110', 'OF-03'}


def _normalize_node_id(node_id):
    return re.sub(r'(OF)(\d+)', r'\1-\2', str(node_id))


def _should_include_nse_node(observed_series, node_id):
    normalized_node_id = _normalize_node_id(node_id)
    if normalized_node_id in _overflow_nodes():
        return True
    return np.std(observed_series) >= 0.01


def _apply_physical_constraints(reconstructed, context, node_indices=None):
    reconstructed = np.maximum(np.asarray(reconstructed, dtype=float), 0.0)
    return reconstructed


def _selected_sensor_measurements(data, sensor_indices, context):
    return (data[sensor_indices, :] - _ctx(context, 'X_mean')[sensor_indices]) / _ctx(context, 'X_std_safe')[sensor_indices]


def _reconstruct_from_measurements(sensor_measurements, sensor_indices, Psi, r, context):
    ind_Psi = np.arange(r)
    Psi_ind = Psi[:, ind_Psi]
    Psi_sensors = Psi[sensor_indices, :][:, ind_Psi]
    reconstructed = Psi_ind @ np.linalg.pinv(Psi_sensors) @ sensor_measurements
    reconstructed = reconstructed * _ctx(context, 'X_std_safe') + _ctx(context, 'X_mean')
    return _apply_physical_constraints(reconstructed, context)


def _nse_values(reconstructed, observed, node_ids=None):
    values = []
    for j in range(observed.shape[0]):
        node_id = node_ids[j] if node_ids is not None and j < len(node_ids) else str(j)
        if not _should_include_nse_node(observed[j, :], node_id):
            continue
        values.append(NSE(reconstructed[j, :], observed[j, :]))
    return values


def _node_nse_map(reconstructed, observed, node_ids):
    node_nse = {}
    for j in range(observed.shape[0]):
        node_id = _normalize_node_id(node_ids[j] if j < len(node_ids) else str(j))
        if not _should_include_nse_node(observed[j, :], node_id):
            continue
        node_nse[node_id] = NSE(reconstructed[j, :], observed[j, :])
    return node_nse


def _overall_event_nse(reconstructed, observed, node_ids):
    included_indices = []
    for j in range(observed.shape[0]):
        node_id = node_ids[j] if j < len(node_ids) else str(j)
        if _should_include_nse_node(observed[j, :], node_id):
            included_indices.append(j)

    if not included_indices:
        return np.nan

    included_indices = np.asarray(included_indices, dtype=int)
    reconstructed_flat = np.asarray(reconstructed[included_indices, :], dtype=float).reshape(-1)
    observed_flat = np.asarray(observed[included_indices, :], dtype=float).reshape(-1)
    return float(NSE(reconstructed_flat, observed_flat))


def _eventwise_dropout_nse_statistics(event_matrices, node_ids, selected_sensors, drop_index, Psi, r, context):
    psi_original = Psi[:, :r]
    remaining_sensors = [sensor for sensor in selected_sensors if sensor != drop_index]
    remaining_node_indices = [idx for idx in range(Psi.shape[0]) if idx != drop_index]
    psi_original_reduced = psi_original[remaining_node_indices, :r]
    psi_reduced = Psi[remaining_sensors, :r]
    pinv_reduced = np.linalg.pinv(psi_reduced)

    baseline_event_means = []
    dropout_event_means = []
    baseline_node_values = defaultdict(list)
    dropout_node_values = defaultdict(list)

    for event_item in event_matrices:
        event_data = event_item['data']
        full_reconstructed = _reconstruct_from_measurements(
            _selected_sensor_measurements(event_data, selected_sensors, context),
            selected_sensors,
            Psi,
            r,
            context
        )
        data_normalized = (event_data - _ctx(context, 'X_mean')) / _ctx(context, 'X_std_safe')
        reconstructed_dropout = (
            psi_original_reduced @ pinv_reduced @ data_normalized[remaining_sensors]
        ) * _ctx(context, 'X_std_safe')[remaining_node_indices] + _ctx(context, 'X_mean')[remaining_node_indices]
        reconstructed_dropout = _apply_physical_constraints(
            reconstructed_dropout,
            context,
            node_indices=remaining_node_indices
        )

        baseline_map = _node_nse_map(
            full_reconstructed[remaining_node_indices],
            event_data[remaining_node_indices],
            [_ctx(context, 'Node_ID')[idx] for idx in remaining_node_indices]
        )
        dropout_map = _node_nse_map(
            reconstructed_dropout,
            event_data[remaining_node_indices],
            [_ctx(context, 'Node_ID')[idx] for idx in remaining_node_indices]
        )

        common_nodes = sorted(set(baseline_map.keys()) & set(dropout_map.keys()))
        if not common_nodes:
            continue

        baseline_event_means.append(float(np.mean([baseline_map[node] for node in common_nodes])))
        dropout_event_means.append(float(np.mean([dropout_map[node] for node in common_nodes])))

        for node in common_nodes:
            baseline_node_values[node].append(float(baseline_map[node]))
            dropout_node_values[node].append(float(dropout_map[node]))

    baseline_mean = float(np.mean(baseline_event_means)) if baseline_event_means else np.nan
    dropout_mean = float(np.mean(dropout_event_means)) if dropout_event_means else np.nan
    baseline_node_mean = {
        node: float(np.mean(values)) for node, values in baseline_node_values.items() if len(values) > 0
    }
    dropout_node_mean = {
        node: float(np.mean(values)) for node, values in dropout_node_values.items() if len(values) > 0
    }
    return baseline_mean, dropout_mean, baseline_node_mean, dropout_node_mean


def _eventwise_dropout_system_level_statistics(event_matrices, node_ids, selected_sensors, drop_index, Psi, r, context):
    remaining_sensors = [sensor for sensor in selected_sensors if sensor != drop_index]
    psi_original = Psi[:, :r]
    psi_reduced = Psi[remaining_sensors, :r]
    pinv_reduced = np.linalg.pinv(psi_reduced)

    baseline_event_values = []
    dropout_event_values = []

    for event_item in event_matrices:
        event_data = event_item['data']
        full_reconstructed = _reconstruct_from_measurements(
            _selected_sensor_measurements(event_data, selected_sensors, context),
            selected_sensors,
            Psi,
            r,
            context
        )
        baseline_event_values.append(float(_overall_event_nse(full_reconstructed, event_data, node_ids)))

        event_normalized = (event_data - _ctx(context, 'X_mean')) / _ctx(context, 'X_std_safe')
        reconstructed_dropout = (psi_original @ pinv_reduced @ event_normalized[remaining_sensors])
        reconstructed_dropout = reconstructed_dropout * _ctx(context, 'X_std_safe') + _ctx(context, 'X_mean')
        reconstructed_dropout = _apply_physical_constraints(reconstructed_dropout, context)
        dropout_event_values.append(float(_overall_event_nse(reconstructed_dropout, event_data, node_ids)))

    baseline_values = [value for value in baseline_event_values if np.isfinite(value)]
    dropout_values = [value for value in dropout_event_values if np.isfinite(value)]
    baseline_mean = float(np.mean(baseline_values)) if baseline_values else np.nan
    dropout_mean = float(np.mean(dropout_values)) if dropout_values else np.nan
    return baseline_mean, dropout_mean, baseline_values, dropout_values


def _selected_sensors_for_r(Psi, r):
    ind_psi = np.arange(r)
    _, _, pivoting = qr(Psi[:, ind_psi].T, pivoting=True)
    return pivoting[:r]


def _ordered_node_values(node_value_map, node_ids):
    ordered_values = []
    for node_id in [_normalize_node_id(node) for node in node_ids]:
        if node_id in node_value_map:
            ordered_values.append(float(node_value_map[node_id]))
    return ordered_values


def _eventwise_node_nse_map_for_sensors(event_matrices, sensor_indices, Psi, r, context):
    event_node_values = defaultdict(list)
    for event_item in event_matrices:
        event_data = event_item['data']
        event_reconstructed = _reconstruct_from_measurements(
            _selected_sensor_measurements(event_data, sensor_indices, context),
            sensor_indices,
            Psi,
            r,
            context
        )
        event_node_map = _node_nse_map(event_reconstructed, event_data, _ctx(context, 'Node_ID'))
        for node_id, nse_value in event_node_map.items():
            event_node_values[node_id].append(float(nse_value))
    return {
        node_id: float(np.mean(values))
        for node_id, values in event_node_values.items()
        if len(values) > 0
    }


def Calculate_eventwise_nse_for_r(event_matrices, node_ids, r, Psi, context):
    sensor_indices = _selected_sensors_for_r(Psi, r)
    node_map = _eventwise_node_nse_map_for_sensors(event_matrices, sensor_indices, Psi, r, context)
    return _ordered_node_values(node_map, node_ids)


def Calculate_eventwise_node_nse_details_for_r(event_matrices, node_ids, r, Psi, context):
    sensor_indices = _selected_sensors_for_r(Psi, r)
    ordered_node_ids = [_normalize_node_id(node_id) for node_id in node_ids]
    event_node_maps = {}

    for event_item in event_matrices:
        event_data = event_item['data']
        event_reconstructed = _reconstruct_from_measurements(
            _selected_sensor_measurements(event_data, sensor_indices, context),
            sensor_indices,
            Psi,
            r,
            context
        )
        event_node_maps[event_item['event_name']] = _node_nse_map(
            event_reconstructed,
            event_data,
            _ctx(context, 'Node_ID')
        )

    mean_node_map = {}
    for node_id in ordered_node_ids:
        values = [
            float(node_map[node_id])
            for node_map in event_node_maps.values()
            if node_id in node_map
        ]
        if values:
            mean_node_map[node_id] = float(np.mean(values))

    return {
        'event_node_maps': event_node_maps,
        'mean_node_map': mean_node_map,
        'ordered_node_ids': [node_id for node_id in ordered_node_ids if node_id in mean_node_map],
    }


def Calculate_eventwise_nse_by_r(event_matrices, node_ids, r_values, Psi, context):
    per_r_node_nse = {}

    for r in r_values:
        per_r_node_nse[r] = _eventwise_node_nse_map_for_sensors(
            event_matrices,
            _selected_sensors_for_r(Psi, r),
            Psi,
            r,
            context
        )

    return per_r_node_nse


def Calculate_eventwise_overall_nse_for_r(event_matrices, node_ids, r, Psi, context):
    sensor_indices = _selected_sensors_for_r(Psi, r)
    event_nse_values = []
    for event_item in event_matrices:
        event_data = event_item['data']
        event_reconstructed = _reconstruct_from_measurements(
            _selected_sensor_measurements(event_data, sensor_indices, context),
            sensor_indices,
            Psi,
            r,
            context
        )
        event_nse_values.append(_overall_event_nse(event_reconstructed, event_data, node_ids))
    return [float(value) for value in event_nse_values if np.isfinite(value)]


def Calculate_eventwise_nse_noisy(
    event_matrices,
    node_ids,
    r,
    noise_level,
    Psi,
    context,
    noise_type="multiplicative",
    num_monte_carlo=1000,
    noise_samples_by_event=None
):
    selected_sensors = _selected_sensors_for_r(Psi, r)
    clean_node_map = _eventwise_node_nse_map_for_sensors(event_matrices, selected_sensors, Psi, r, context)
    clean_values = _ordered_node_values(clean_node_map, node_ids)

    noisy_node_maps = [defaultdict(list) for _ in range(num_monte_carlo)]

    for event_item in event_matrices:
        event_name = event_item['event_name']
        event_data = event_item['data']
        clean_sensor_measurements = _selected_sensor_measurements(event_data, selected_sensors, context)

        event_noise_samples = None
        if noise_samples_by_event is not None:
            event_noise_samples = noise_samples_by_event.get(event_name)

        if event_noise_samples is None:
            std_dev = noise_level / 3.0
            if noise_type == "multiplicative":
                event_noise_samples = np.random.normal(
                    0,
                    std_dev,
                    size=(num_monte_carlo, len(selected_sensors), event_data.shape[1])
                )
            elif noise_type == "additive":
                sensor_std = np.std(clean_sensor_measurements, axis=1, keepdims=True)
                event_noise_samples = np.random.normal(
                    0,
                    noise_level * sensor_std,
                    size=(num_monte_carlo, len(selected_sensors), event_data.shape[1])
                )
            else:
                event_noise_samples = np.zeros((num_monte_carlo, len(selected_sensors), event_data.shape[1]))

        for mc_iter in range(num_monte_carlo):
            noise_matrix = event_noise_samples[mc_iter]
            if noise_type == "multiplicative":
                noisy_sensor_measurements = clean_sensor_measurements * (1 + noise_matrix)
            elif noise_type == "additive":
                noisy_sensor_measurements = clean_sensor_measurements + noise_matrix
            else:
                noisy_sensor_measurements = clean_sensor_measurements.copy()

            reconstructed_noisy = _reconstruct_from_measurements(
                noisy_sensor_measurements,
                selected_sensors,
                Psi,
                r,
                context
            )
            noisy_event_map = _node_nse_map(reconstructed_noisy, event_data, _ctx(context, 'Node_ID'))
            for node_id, nse_value in noisy_event_map.items():
                noisy_node_maps[mc_iter][node_id].append(float(nse_value))

    noisy_values = []
    normalized_node_ids = [_normalize_node_id(node) for node in node_ids]
    for mc_iter in range(num_monte_carlo):
        per_iteration_values = []
        for node_id in normalized_node_ids:
            if node_id in noisy_node_maps[mc_iter]:
                per_iteration_values.append(float(np.mean(noisy_node_maps[mc_iter][node_id])))
        noisy_values.append(per_iteration_values)

    return {
        'pivot': selected_sensors,
        'nse_clean': clean_values,
        'nse_noisy': noisy_values,
    }


def Calculate_eventwise_overall_nse_noisy(
    event_matrices,
    node_ids,
    r,
    noise_level,
    Psi,
    context,
    noise_type="multiplicative",
    num_monte_carlo=1000,
    noise_samples_by_event=None
):
    selected_sensors = _selected_sensors_for_r(Psi, r)
    clean_event_values = []
    noisy_event_values = [[] for _ in range(num_monte_carlo)]

    for event_item in event_matrices:
        event_name = event_item['event_name']
        event_data = event_item['data']
        clean_sensor_measurements = _selected_sensor_measurements(event_data, selected_sensors, context)

        clean_reconstructed = _reconstruct_from_measurements(
            clean_sensor_measurements,
            selected_sensors,
            Psi,
            r,
            context
        )
        clean_event_values.append(float(_overall_event_nse(clean_reconstructed, event_data, node_ids)))

        event_noise_samples = None
        if noise_samples_by_event is not None:
            event_noise_samples = noise_samples_by_event.get(event_name)

        if event_noise_samples is None:
            std_dev = noise_level / 3.0
            if noise_type == "multiplicative":
                event_noise_samples = np.random.normal(
                    0,
                    std_dev,
                    size=(num_monte_carlo, len(selected_sensors), event_data.shape[1])
                )
            elif noise_type == "additive":
                sensor_std = np.std(clean_sensor_measurements, axis=1, keepdims=True)
                event_noise_samples = np.random.normal(
                    0,
                    noise_level * sensor_std,
                    size=(num_monte_carlo, len(selected_sensors), event_data.shape[1])
                )
            else:
                event_noise_samples = np.zeros((num_monte_carlo, len(selected_sensors), event_data.shape[1]))

        for mc_iter in range(num_monte_carlo):
            noise_matrix = event_noise_samples[mc_iter]
            if noise_type == "multiplicative":
                noisy_sensor_measurements = clean_sensor_measurements * (1 + noise_matrix)
            elif noise_type == "additive":
                noisy_sensor_measurements = clean_sensor_measurements + noise_matrix
            else:
                noisy_sensor_measurements = clean_sensor_measurements.copy()

            reconstructed_noisy = _reconstruct_from_measurements(
                noisy_sensor_measurements,
                selected_sensors,
                Psi,
                r,
                context
            )
            noisy_event_values[mc_iter].append(float(_overall_event_nse(reconstructed_noisy, event_data, node_ids)))

    return {
        'pivot': selected_sensors,
        'nse_clean': [value for value in clean_event_values if np.isfinite(value)],
        'nse_noisy': [
            [value for value in iteration_values if np.isfinite(value)]
            for iteration_values in noisy_event_values
        ],
    }


def _build_exhaustive_payload(data, r, Psi, context):
    raw_data = np.asarray(data, dtype=float)
    payload = {
        'r': r,
        'Psi': np.asarray(Psi, dtype=float),
        'Psi_ind': np.asarray(Psi[:, :r], dtype=float),
        'raw_data': raw_data,
        'raw_data_mean': np.mean(raw_data, axis=0, keepdims=True),
        'Node_ID': _ctx(context, 'Node_ID'),
        'X_max_limits': np.asarray(_ctx(context, 'X_max_limits'), dtype=float),
        'X_mean_sq': np.asarray(_ctx(context, 'X_mean'), dtype=float),
        'X_std_sq': np.asarray(_ctx(context, 'X_std_safe'), dtype=float),
    }
    payload['denominator_per_timestep'] = np.sum((raw_data - payload['raw_data_mean']) ** 2, axis=0)
    payload['zero_den_mask'] = payload['denominator_per_timestep'] == 0
    return payload


def _evaluate_exhaustive_combo(sensor_indices, payload):
    r = payload['r']
    Psi_combo = payload['Psi'][sensor_indices, :][:, :r]
    try:
        y_normalized = (payload['raw_data'][sensor_indices, :] - payload['X_mean_sq'][sensor_indices]) / payload['X_std_sq'][sensor_indices]
        coeffs = np.linalg.solve(Psi_combo, y_normalized)
        x_hat_physical = (payload['Psi_ind'] @ coeffs) * payload['X_std_sq'] + payload['X_mean_sq']
    except np.linalg.LinAlgError:
        return None

    x_hat_physical = _apply_physical_constraints(x_hat_physical, payload)
    mean_nse = float(np.mean(_nse_values(x_hat_physical, payload['raw_data'], payload['Node_ID'])))
    return mean_nse, sensor_indices.tolist()


_EXHAUSTIVE_WORKER_PAYLOAD = None
_EVENTWISE_EXHAUSTIVE_WORKER_PAYLOAD = None


def _init_exhaustive_worker(payload):
    global _EXHAUSTIVE_WORKER_PAYLOAD
    _EXHAUSTIVE_WORKER_PAYLOAD = payload


def _build_eventwise_exhaustive_payload(event_matrices, r, Psi, context):
    return {
        'r': r,
        'Psi': np.asarray(Psi, dtype=float),
        'Psi_ind': np.asarray(Psi[:, :r], dtype=float),
        'event_matrices': [
            {
                'event_name': event_item['event_name'],
                'data': np.asarray(event_item['data'], dtype=float),
            }
            for event_item in event_matrices
        ],
        'Node_ID': _ctx(context, 'Node_ID'),
        'X_max_limits': np.asarray(_ctx(context, 'X_max_limits'), dtype=float),
        'X_mean_sq': np.asarray(_ctx(context, 'X_mean'), dtype=float),
        'X_std_sq': np.asarray(_ctx(context, 'X_std_safe'), dtype=float),
    }


def Calculate_eventwise_overall_nse_for_sensors(event_matrices, node_ids, sensor_indices, Psi, r, context):
    event_nse_values = []
    for event_item in event_matrices:
        event_data = event_item['data']
        event_reconstructed = _reconstruct_from_measurements(
            _selected_sensor_measurements(event_data, sensor_indices, context),
            sensor_indices,
            Psi,
            r,
            context
        )
        event_nse_values.append(_overall_event_nse(event_reconstructed, event_data, node_ids))
    return [float(value) for value in event_nse_values if np.isfinite(value)]


def _evaluate_eventwise_exhaustive_combo(sensor_indices, payload):
    r = payload['r']
    Psi_combo = payload['Psi'][sensor_indices, :][:, :r]
    event_values = []

    try:
        for event_item in payload['event_matrices']:
            event_data = event_item['data']
            y_normalized = (
                event_data[sensor_indices, :] - payload['X_mean_sq'][sensor_indices]
            ) / payload['X_std_sq'][sensor_indices]
            coeffs = np.linalg.solve(Psi_combo, y_normalized)
            x_hat_physical = (payload['Psi_ind'] @ coeffs) * payload['X_std_sq'] + payload['X_mean_sq']
            x_hat_physical = _apply_physical_constraints(x_hat_physical, payload)
            event_values.append(float(_overall_event_nse(x_hat_physical, event_data, payload['Node_ID'])))
    except np.linalg.LinAlgError:
        return None

    finite_values = [value for value in event_values if np.isfinite(value)]
    if not finite_values:
        return None
    return float(np.mean(finite_values)), sensor_indices.tolist(), event_values


def _init_eventwise_exhaustive_worker(payload):
    global _EVENTWISE_EXHAUSTIVE_WORKER_PAYLOAD
    _EVENTWISE_EXHAUSTIVE_WORKER_PAYLOAD = payload


def _eventwise_exhaustive_worker_for_prefix(first_idx):
    payload = _EVENTWISE_EXHAUSTIVE_WORKER_PAYLOAD
    r = payload['r']
    N = payload['Psi'].shape[0]
    valid_count = 0
    total_count = math.comb(N - first_idx - 1, r - 1) if r > 1 else 1
    best_mean_nse = None
    best_sensors = None
    best_event_values = None
    pooled_event_values = []

    if r == 1:
        result = _evaluate_eventwise_exhaustive_combo(np.asarray([first_idx], dtype=int), payload)
        if result is not None:
            best_mean_nse, best_sensors, best_event_values = result
            pooled_event_values.extend(best_event_values)
            valid_count = 1
        return {
            'prefix': first_idx,
            'processed_total': total_count,
            'processed_valid': valid_count,
            'best_mean_nse': best_mean_nse,
            'best_sensors': best_sensors,
            'best_event_values': best_event_values,
            'pooled_event_values': pooled_event_values,
        }

    for suffix in combinations(range(first_idx + 1, N), r - 1):
        sensor_indices = np.asarray((first_idx, *suffix), dtype=int)
        result = _evaluate_eventwise_exhaustive_combo(sensor_indices, payload)
        if result is None:
            continue
        valid_count += 1
        mean_nse, sensors, event_values = result
        pooled_event_values.extend(event_values)
        if best_mean_nse is None or mean_nse > best_mean_nse:
            best_mean_nse = mean_nse
            best_sensors = sensors
            best_event_values = event_values

    return {
        'prefix': first_idx,
        'processed_total': total_count,
        'processed_valid': valid_count,
        'best_mean_nse': best_mean_nse,
        'best_sensors': best_sensors,
        'best_event_values': best_event_values,
        'pooled_event_values': pooled_event_values,
    }


def _exhaustive_worker_for_prefix(first_idx):
    payload = _EXHAUSTIVE_WORKER_PAYLOAD
    r = payload['r']
    N = payload['Psi'].shape[0]
    valid_count = 0
    total_count = math.comb(N - first_idx - 1, r - 1) if r > 1 else 1
    best_mean_nse = None
    best_sensors = None

    if r == 1:
        result = _evaluate_exhaustive_combo(np.asarray([first_idx], dtype=int), payload)
        if result is not None:
            best_mean_nse, best_sensors = result
            valid_count = 1
        return {
            'prefix': first_idx,
            'processed_total': total_count,
            'processed_valid': valid_count,
            'best_mean_nse': best_mean_nse,
            'best_sensors': best_sensors,
        }

    for suffix in combinations(range(first_idx + 1, N), r - 1):
        sensor_indices = np.asarray((first_idx, *suffix), dtype=int)
        result = _evaluate_exhaustive_combo(sensor_indices, payload)
        if result is None:
            continue
        valid_count += 1
        mean_nse, sensors = result
        if best_mean_nse is None or mean_nse > best_mean_nse:
            best_mean_nse = mean_nse
            best_sensors = sensors

    return {
        'prefix': first_idx,
        'processed_total': total_count,
        'processed_valid': valid_count,
        'best_mean_nse': best_mean_nse,
        'best_sensors': best_sensors,
    }


def SVDQR_NSE(data, r, Psi, context):
    ind_Psi = np.arange(r)
    Psi_ind = Psi[:, ind_Psi]

    _, _, pivoting = qr(Psi_ind.T, pivoting=True)
    Pivot = pivoting[:r]
    Psi_pivot = Psi[Pivot, :][:, ind_Psi]
    Psi_pivot_pinv = np.linalg.pinv(Psi_pivot)

    data_normalized = (data - _ctx(context, 'X_mean')) / _ctx(context, 'X_std_safe')
    all_x_ph = (Psi_ind @ Psi_pivot_pinv @ data_normalized[Pivot]) * _ctx(context, 'X_std_safe') + _ctx(context, 'X_mean')
    all_x_ph = _apply_physical_constraints(all_x_ph, context)
    return _nse_values(all_x_ph, data, _ctx(context, 'Node_ID')), [all_x_ph[:, i] for i in range(data.shape[1])]


def SVDQR_NSE_Physical(data, r, Psi, context):
    ind_Psi = np.arange(r)
    Psi_ind = Psi[:, ind_Psi]
    _, _, pivoting = qr(Psi_ind.T, pivoting=True)
    Pivot = pivoting[:r]
    Psi_pivot = Psi[Pivot, :][:, ind_Psi]
    Psi_pivot_pinv = np.linalg.pinv(Psi_pivot)

    data_normalized = (data - _ctx(context, 'X_mean')) / _ctx(context, 'X_std_safe')
    all_x_normalized = Psi_ind @ Psi_pivot_pinv @ data_normalized[Pivot]
    all_x_physical = all_x_normalized * _ctx(context, 'X_std_safe') + _ctx(context, 'X_mean')
    all_x_physical = _apply_physical_constraints(all_x_physical, context)
    return _nse_values(all_x_physical, data, _ctx(context, 'Node_ID')), [all_x_physical[:, i] for i in range(data.shape[1])]


def SVDQR_NSE_Nodewise(data, r, Psi, context):
    ind_Psi = np.arange(r)
    Psi_ind = Psi[:, ind_Psi]
    _, _, pivoting = qr(Psi_ind.T, pivoting=True)
    Pivot = pivoting[:r]
    Psi_pivot = Psi[Pivot, :][:, ind_Psi]
    Psi_pivot_pinv = np.linalg.pinv(Psi_pivot)

    data_normalized = (data - _ctx(context, 'X_mean')) / _ctx(context, 'X_std_safe')
    all_x_normalized = Psi_ind @ Psi_pivot_pinv @ data_normalized[Pivot]
    all_x_physical = all_x_normalized * _ctx(context, 'X_std_safe') + _ctx(context, 'X_mean')
    all_x_physical = _apply_physical_constraints(all_x_physical, context)
    return _nse_values(all_x_physical, data, _ctx(context, 'Node_ID'))


def SVDQR_NSE_Noisy(
    data,
    r,
    noise_level,
    Psi,
    context,
    noise_type="multiplicative",
    num_monte_carlo=1000,
    noise_samples=None
):
    ind_Psi = np.arange(r)
    Psi_ind = Psi[:, ind_Psi]

    _, _, pivoting = qr(Psi_ind.T, pivoting=True)
    Pivot = pivoting[:r]
    clean_sensor_measurements = _selected_sensor_measurements(data, Pivot, context)
    all_x_physical = _reconstruct_from_measurements(clean_sensor_measurements, Pivot, Psi, r, context)
    nse_values = _nse_values(all_x_physical, data, _ctx(context, 'Node_ID'))

    nse_values_noisy = []
    std_dev = noise_level / 3.0
    if noise_samples is None:
        if noise_type == "multiplicative":
            noise_samples = np.random.normal(0, std_dev, size=(num_monte_carlo, len(Pivot), data.shape[1]))
        elif noise_type == "additive":
            sensor_std = np.std(clean_sensor_measurements, axis=1, keepdims=True)
            noise_samples = np.random.normal(0, noise_level * sensor_std, size=(num_monte_carlo, len(Pivot), data.shape[1]))
        else:
            noise_samples = np.zeros((num_monte_carlo, len(Pivot), data.shape[1]))

    for mc_iter in range(num_monte_carlo):
        noise_matrix = noise_samples[mc_iter]
        if noise_type == "multiplicative":
            noisy_sensor_measurements = clean_sensor_measurements * (1 + noise_matrix)
        elif noise_type == "additive":
            noisy_sensor_measurements = clean_sensor_measurements + noise_matrix
        else:
            noisy_sensor_measurements = clean_sensor_measurements.copy()

        x_hat_noisy_physical = _reconstruct_from_measurements(noisy_sensor_measurements, Pivot, Psi, r, context)
        nse_values_noisy.append(_nse_values(x_hat_noisy_physical, data, _ctx(context, 'Node_ID')))

    return {
        'pivot': Pivot,
        'nse_clean': nse_values,
        'nse_noisy': nse_values_noisy,
        'noise_samples': noise_samples,
    }


def SVDQR_NSE_SD(r, Psi, context, event_matrices):
    nse_prediction_d = {}
    event_nse_prediction_d = {}
    Node_ID_D = {}

    ind_Psi = np.arange(r)
    _, _, pivoting = qr(Psi[:, ind_Psi].T, pivoting=True)
    Pivot = pivoting[:r]
    Psi_r = Psi[:, ind_Psi]
    full_node_values = defaultdict(list)
    full_event_values = []
    for event_item in event_matrices:
        event_data = event_item['data']
        event_reconstructed = _reconstruct_from_measurements(
            _selected_sensor_measurements(event_data, Pivot, context),
            Pivot,
            Psi,
            r,
            context
        )
        full_event_values.append(float(_overall_event_nse(event_reconstructed, event_data, _ctx(context, 'Node_ID'))))
        event_node_map = _node_nse_map(event_reconstructed, event_data, _ctx(context, 'Node_ID'))
        for node_id, nse_value in event_node_map.items():
            full_node_values[node_id].append(float(nse_value))
    nse_prediction_d[r] = [
        float(np.mean(full_node_values[node_id]))
        for node_id in [_normalize_node_id(node) for node in _ctx(context, 'Node_ID')]
        if node_id in full_node_values
    ]
    event_nse_prediction_d[r] = [value for value in full_event_values if np.isfinite(value)]
    Node_ID_D[r] = "Full_set"

    for drop_index in range(len(Pivot)):
        current_drop = Pivot[drop_index]
        remaining_Pivot = np.delete(Pivot, drop_index)
        Psi_remaining = Psi[remaining_Pivot, :][:, ind_Psi]
        Psi_remaining_pinv = np.linalg.pinv(Psi_remaining)

        reduced_indices = np.delete(np.arange(Psi.shape[0]), current_drop)
        dropout_node_values = defaultdict(list)
        dropout_event_values = []
        for event_item in event_matrices:
            event_data = event_item['data']
            event_normalized = (event_data - _ctx(context, 'X_mean')) / _ctx(context, 'X_std_safe')
            event_reconstructed_drop = (
                Psi_r @ Psi_remaining_pinv @ event_normalized[remaining_Pivot]
            ) * _ctx(context, 'X_std_safe') + _ctx(context, 'X_mean')
            event_reconstructed_drop = _apply_physical_constraints(event_reconstructed_drop, context)
            event_x_hat_drop = np.delete(event_reconstructed_drop, current_drop, axis=0)
            event_y_drop = np.delete(event_data, current_drop, axis=0)
            dropout_event_values.append(float(_overall_event_nse(
                _apply_physical_constraints(event_x_hat_drop, context, node_indices=reduced_indices),
                event_y_drop,
                [_ctx(context, 'Node_ID')[idx] for idx in reduced_indices]
            )))
            event_node_map = _node_nse_map(
                _apply_physical_constraints(event_x_hat_drop, context, node_indices=reduced_indices),
                event_y_drop,
                [_ctx(context, 'Node_ID')[idx] for idx in reduced_indices]
            )
            for node_id, nse_value in event_node_map.items():
                dropout_node_values[node_id].append(float(nse_value))

        nse_prediction_d[drop_index] = [
            float(np.mean(dropout_node_values[node_id]))
            for node_id in [_normalize_node_id(_ctx(context, 'Node_ID')[idx]) for idx in reduced_indices]
            if node_id in dropout_node_values
        ]
        event_nse_prediction_d[drop_index] = [value for value in dropout_event_values if np.isfinite(value)]

        node_id = _ctx(context, 'Node_ID')[int(current_drop)]
        Node_ID_D[drop_index] = re.sub(r'(OF)(\d+)', r'\1-\2', node_id)
    return nse_prediction_d, event_nse_prediction_d, Node_ID_D


def SVDQR_NSE_Random(data, r, Psi, context, num_iterations=100000):
    nse_values_random = []
    ind_Psi = np.arange(r)
    Psi_ind = Psi[:, ind_Psi]
    all_indices = np.arange(data.shape[0])

    for _ in range(num_iterations):
        random_indices = np.random.choice(all_indices, size=r, replace=False)

        Psi_random = Psi[random_indices, :][:, ind_Psi]
        Psi_random_pinv = np.linalg.pinv(Psi_random)
        nse_values_random_iter = []
        for i in range(data.shape[1]):
            y = data[:, i:i + 1]
            y_normalized = (y[random_indices] - _ctx(context, 'X_mean')[random_indices]) / _ctx(context, 'X_std_safe')[random_indices]
            x_hat_physical = (Psi_ind @ Psi_random_pinv @ y_normalized) * _ctx(context, 'X_std_safe') + _ctx(context, 'X_mean')
            x_hat_physical = _apply_physical_constraints(x_hat_physical, context)
            nse_values_random_iter.append(float(np.mean(_nse_values(x_hat_physical, y, [_ctx(context, 'Node_ID')[i]]))))
        nse_values_random.append(nse_values_random_iter)
    return nse_values_random


def SVDQR_NSE_Random_Eventwise(event_matrices, node_ids, r, Psi, context, num_iterations=100000):
    nse_values_random = []
    all_indices = np.arange(Psi.shape[0])

    for _ in range(num_iterations):
        random_indices = np.random.choice(all_indices, size=r, replace=False)

        node_map = _eventwise_node_nse_map_for_sensors(event_matrices, random_indices, Psi, r, context)
        nse_values_random.append(_ordered_node_values(node_map, node_ids))

    return nse_values_random


def SVDQR_NSE_Random_Eventwise_Overall(event_matrices, node_ids, r, Psi, context, num_iterations=100000):
    nse_values_random = []
    all_indices = np.arange(Psi.shape[0])

    for _ in range(num_iterations):
        random_indices = np.random.choice(all_indices, size=r, replace=False)
        nse_values_random.append(
            Calculate_eventwise_overall_nse_for_sensors(
                event_matrices,
                node_ids,
                random_indices,
                Psi,
                r,
                context
            )
        )

    return nse_values_random


def SVDQR_NSE_Exhaustive_Eventwise_Overall(event_matrices, node_ids, r, Psi, context, max_combos=2000000, num_workers=1):
    N = Psi.shape[0]
    total_combos = math.comb(N, r)
    print(f"Total combinations to evaluate for system-level exhaustive benchmark for r={r}: {total_combos}")
    if total_combos > max_combos:
        print(f"Combinations ({total_combos}) exceed max limit ({max_combos}). Skipping exhaustive benchmark.")
        return None, None, None, None, None

    payload = _build_eventwise_exhaustive_payload(event_matrices, r, Psi, context)
    best_mean_nse = None
    best_sensors = None
    best_event_values = None
    evaluated_combos = 0
    pooled_event_values = []
    start_time = time.perf_counter()
    progress_interval = max(1, min(50000, total_combos // 20 if total_combos > 20 else 1))

    if num_workers is None or num_workers < 1:
        num_workers = 1

    if num_workers == 1:
        for combo_index, combo in enumerate(combinations(range(N), r), start=1):
            sensor_indices = np.fromiter(combo, dtype=int, count=r)
            result = _evaluate_eventwise_exhaustive_combo(sensor_indices, payload)
            if result is None:
                continue
            evaluated_combos += 1
            mean_nse, sensors, event_values = result
            pooled_event_values.extend(event_values)
            if best_mean_nse is None or mean_nse > best_mean_nse:
                best_mean_nse = mean_nse
                best_sensors = sensors
                best_event_values = event_values
            if combo_index % progress_interval == 0 or combo_index == total_combos:
                elapsed = time.perf_counter() - start_time
                rate = combo_index / elapsed if elapsed > 0 else 0.0
                remaining = (total_combos - combo_index) / rate if rate > 0 else float('inf')
                print(
                    f"Progress: {combo_index}/{total_combos} ({combo_index / total_combos * 100:.1f}%) | "
                    f"Elapsed: {elapsed:.1f}s | ETA: {remaining:.1f}s | "
                    f"Current best mean system-level NSE: {best_mean_nse:.4f}"
                )
    else:
        prefix_indices = list(range(0, N - r + 1))
        processed_total = 0
        try:
            with mp.Pool(processes=num_workers, initializer=_init_eventwise_exhaustive_worker, initargs=(payload,)) as pool:
                for worker_result in pool.imap_unordered(_eventwise_exhaustive_worker_for_prefix, prefix_indices):
                    processed_total += worker_result['processed_total']
                    evaluated_combos += worker_result['processed_valid']
                    pooled_event_values.extend(worker_result['pooled_event_values'])
                    worker_best = worker_result['best_mean_nse']
                    if worker_best is not None and (best_mean_nse is None or worker_best > best_mean_nse):
                        best_mean_nse = worker_best
                        best_sensors = worker_result['best_sensors']
                        best_event_values = worker_result['best_event_values']
                    elapsed = time.perf_counter() - start_time
                    rate = processed_total / elapsed if elapsed > 0 else 0.0
                    remaining = (total_combos - processed_total) / rate if rate > 0 else float('inf')
                    print(
                        f"Progress: {processed_total}/{total_combos} ({processed_total / total_combos * 100:.1f}%) | "
                        f"Elapsed: {elapsed:.1f}s | ETA: {remaining:.1f}s | "
                        f"Current best mean system-level NSE: {best_mean_nse:.4f}"
                    )
        except Exception as exc:
            print(f"Parallel system-level exhaustive setup failed ({exc}). Falling back to serial execution.")
            return SVDQR_NSE_Exhaustive_Eventwise_Overall(
                event_matrices, node_ids, r, Psi, context, max_combos=max_combos, num_workers=1
            )

    total_elapsed = time.perf_counter() - start_time
    print(
        f"System-level exhaustive best mean NSE for r={r}: {best_mean_nse:.4f} | "
        f"Sensors: {best_sensors}"
    )
    print(f"System-level exhaustive search evaluated {evaluated_combos} valid combinations in {total_elapsed:.2f}s")
    return pooled_event_values, best_event_values, best_sensors, evaluated_combos, total_elapsed


def SVDQR_NSE_Exhaustive(data, r, Psi, context, max_combos=10000000, num_workers=1):
    N = Psi.shape[0]
    total_combos = math.comb(N, r)
    print(f"Total combinations to evaluate for r={r}: {total_combos}")
    if total_combos > max_combos:
        print(f"Combinations ({total_combos}) exceed max limit ({max_combos}). Skipping exhaustive search.")
        return None, None, None, None, None

    best_mean_nse = None
    best_sensors = None
    start_time = time.perf_counter()
    evaluated_combos = 0
    progress_interval = max(1, min(50000, total_combos // 20 if total_combos > 20 else 1))
    payload = _build_exhaustive_payload(data, r, Psi, context)

    if num_workers is None or num_workers < 1:
        num_workers = 1

    if num_workers == 1:
        for combo_index, combo in enumerate(combinations(range(N), r), start=1):
            sensor_indices = np.fromiter(combo, dtype=int, count=r)
            result = _evaluate_exhaustive_combo(sensor_indices, payload)
            if result is None:
                continue
            evaluated_combos += 1
            mean_nse, sensors = result
            if best_mean_nse is None or mean_nse > best_mean_nse:
                best_mean_nse = mean_nse
                best_sensors = sensors
            if combo_index % progress_interval == 0 or combo_index == total_combos:
                elapsed = time.perf_counter() - start_time
                rate = combo_index / elapsed if elapsed > 0 else 0.0
                remaining = (total_combos - combo_index) / rate if rate > 0 else float('inf')
                print(
                    f"Progress: {combo_index}/{total_combos} ({combo_index / total_combos * 100:.1f}%) | "
                    f"Elapsed: {elapsed:.1f}s | ETA: {remaining:.1f}s | "
                    f"Current best NSE: {best_mean_nse:.4f}"
                )
    else:
        prefix_indices = list(range(0, N - r + 1))
        processed_total = 0
        try:
            with mp.Pool(processes=num_workers, initializer=_init_exhaustive_worker, initargs=(payload,)) as pool:
                for worker_result in pool.imap_unordered(_exhaustive_worker_for_prefix, prefix_indices):
                    processed_total += worker_result['processed_total']
                    evaluated_combos += worker_result['processed_valid']
                    worker_best = worker_result['best_mean_nse']
                    if worker_best is not None and (best_mean_nse is None or worker_best > best_mean_nse):
                        best_mean_nse = worker_best
                        best_sensors = worker_result['best_sensors']
                    elapsed = time.perf_counter() - start_time
                    rate = processed_total / elapsed if elapsed > 0 else 0.0
                    remaining = (total_combos - processed_total) / rate if rate > 0 else float('inf')
                    print(
                        f"Progress: {processed_total}/{total_combos} ({processed_total / total_combos * 100:.1f}%) | "
                        f"Elapsed: {elapsed:.1f}s | ETA: {remaining:.1f}s | "
                        f"Current best NSE: {best_mean_nse:.4f}"
                    )
        except Exception as exc:
            print(f"Parallel exhaustive search setup failed ({exc}). Falling back to serial execution.")
            num_workers = 1
            for combo_index, combo in enumerate(combinations(range(N), r), start=1):
                sensor_indices = np.fromiter(combo, dtype=int, count=r)
                result = _evaluate_exhaustive_combo(sensor_indices, payload)
                if result is None:
                    continue
                evaluated_combos += 1
                mean_nse, sensors = result
                if best_mean_nse is None or mean_nse > best_mean_nse:
                    best_mean_nse = mean_nse
                    best_sensors = sensors
                if combo_index % progress_interval == 0 or combo_index == total_combos:
                    elapsed = time.perf_counter() - start_time
                    rate = combo_index / elapsed if elapsed > 0 else 0.0
                    remaining = (total_combos - combo_index) / rate if rate > 0 else float('inf')
                    print(
                        f"Progress: {combo_index}/{total_combos} ({combo_index / total_combos * 100:.1f}%) | "
                        f"Elapsed: {elapsed:.1f}s | ETA: {remaining:.1f}s | "
                        f"Current best NSE: {best_mean_nse:.4f}"
                    )
    total_elapsed = time.perf_counter() - start_time
    print(
        f"Exhaustive Search Best Mean NSE for r={r}: {best_mean_nse:.4f} | "
        f"Sensors: {best_sensors}"
    )
    print(f"Exhaustive search evaluated {evaluated_combos} valid combinations in {total_elapsed:.2f}s")
    return best_mean_nse, best_sensors, evaluated_combos, total_elapsed


def SVDQR(X, Psi, S, r):
    ind_Psi = np.arange(r)
    _, _, pivoting = qr(Psi[:, ind_Psi].T, pivoting=True)
    Pivot = pivoting[:r]
    Psi_selected = Psi[Pivot, :][:, ind_Psi]

    n_rows = len(Pivot)
    residuals = np.zeros(n_rows)
    relative_residuals = np.zeros(n_rows)

    for i in range(n_rows):
        current_row = Psi_selected[i, :]
        other_rows = np.delete(Psi_selected, i, axis=0)
        if other_rows.shape[0] > 0:
            projection_of_row = other_rows.T @ np.linalg.pinv(other_rows @ other_rows.T) @ other_rows @ current_row
            residual_vector = current_row - projection_of_row
            residual_norm = norm(residual_vector)
            residuals[i] = residual_norm
            row_norm = norm(current_row)
            relative_residuals[i] = residual_norm / row_norm if row_norm > 0 else 0.0
        else:
            residuals[i] = norm(current_row)
            relative_residuals[i] = 1.0
    return Pivot, residuals, relative_residuals


def Calculate_residuals_NSE(data, Node_ID, r_values, Psi, context):
    residuals_nse = []
    residuals_nse_mean = {}
    node_data = defaultdict(list)

    for r in r_values:
        ind_Psi = np.arange(r)
        _, _, pivoting = qr(Psi[:, ind_Psi].T, pivoting=True)
        selected_sensors = pivoting[:r]
        Psi_original = Psi[:, :r]

        for _, drop_index in enumerate(selected_sensors):
            remaining_sensors = [s for s in selected_sensors if s != drop_index]
            Psi_remaining_sensors = [s for s in range(Psi_original.shape[0]) if s != drop_index]
            Psi_original_red = Psi_original[Psi_remaining_sensors, :r]
            Psi_red = Psi[remaining_sensors, :r]
            pinv_red = np.linalg.pinv(Psi_red)

            data_normalized = (data - _ctx(context, 'X_mean')) / _ctx(context, 'X_std_safe')
            all_x_physical_d = (Psi_original_red @ pinv_red @ data_normalized[remaining_sensors]) * _ctx(context, 'X_std_safe')[Psi_remaining_sensors] + _ctx(context, 'X_mean')[Psi_remaining_sensors]
            all_x_physical_d = _apply_physical_constraints(
                all_x_physical_d,
                context,
                node_indices=Psi_remaining_sensors
            )
            avg_nse = np.mean(_nse_values(
                all_x_physical_d,
                data[Psi_remaining_sensors],
                [Node_ID[idx] for idx in Psi_remaining_sensors]
            ))
            current_row = Psi_original[drop_index, :]
            other_rows = Psi_original[remaining_sensors, :]

            if other_rows.shape[0] > 0:
                projection_matrix = other_rows.T @ np.linalg.pinv(other_rows @ other_rows.T) @ other_rows
                projection_of_row = projection_matrix @ current_row.T
                residual_vector = current_row.T - projection_of_row
                residual_norm = norm(residual_vector)
                row_norm = norm(current_row.T)
                relative_residual = residual_norm / row_norm if row_norm > 0 else 0
            else:
                relative_residual = 1.0

            residuals_nse.append((relative_residual, avg_nse, Node_ID[drop_index]))
            node_data[Node_ID[drop_index]].append((avg_nse, relative_residual))

    for node_id, values in node_data.items():
        nse_list, relative_residual_list = zip(*values)
        mean_avg_nse = sum(nse_list) / len(nse_list)
        mean_relative_residual = sum(relative_residual_list) / len(relative_residual_list)
        std_avg_nse = np.std(nse_list, ddof=1) if len(nse_list) > 1 else 0
        std_relative_residual = np.std(relative_residual_list, ddof=1) if len(relative_residual_list) > 1 else 0
        residuals_nse_mean[node_id] = (mean_avg_nse, mean_relative_residual, std_avg_nse, std_relative_residual)
    return residuals_nse, residuals_nse_mean


def Calculate_dropout_delta_NSE(Node_ID, r, Psi, context, event_matrices):
    dropout_delta_nse = []
    dropout_delta_nse_mean = {}
    node_data = defaultdict(list)

    ind_Psi = np.arange(r)
    _, _, pivoting = qr(Psi[:, ind_Psi].T, pivoting=True)
    selected_sensors = pivoting[:r]

    Psi_original = Psi[:, :r]
    for drop_index in selected_sensors:
        remaining_sensors = [s for s in selected_sensors if s != drop_index]
        remaining_node_indices = [idx for idx in range(Psi_original.shape[0]) if idx != drop_index]
        baseline_mean_nse, dropout_mean_nse, _, _ = _eventwise_dropout_nse_statistics(
            event_matrices, Node_ID, selected_sensors, drop_index, Psi, r, context
        )
        delta_mean_nse = baseline_mean_nse - dropout_mean_nse

        current_row = Psi_original[drop_index, :]
        other_rows = Psi_original[remaining_sensors, :]
        if other_rows.shape[0] > 0:
            projection_matrix = other_rows.T @ np.linalg.pinv(other_rows @ other_rows.T) @ other_rows
            projection_of_row = projection_matrix @ current_row.T
            residual_vector = current_row.T - projection_of_row
            residual_norm = norm(residual_vector)
            row_norm = norm(current_row.T)
            relative_residual = residual_norm / row_norm if row_norm > 0 else 0.0
        else:
            relative_residual = 1.0

        dropout_delta_nse.append((relative_residual, delta_mean_nse, Node_ID[drop_index]))
        node_data[Node_ID[drop_index]].append((delta_mean_nse, relative_residual))

    for node_id, values in node_data.items():
        delta_nse_list, relative_residual_list = zip(*values)
        mean_delta_nse = sum(delta_nse_list) / len(delta_nse_list)
        mean_relative_residual = sum(relative_residual_list) / len(relative_residual_list)
        std_delta_nse = np.std(delta_nse_list, ddof=1) if len(delta_nse_list) > 1 else 0.0
        std_relative_residual = np.std(relative_residual_list, ddof=1) if len(relative_residual_list) > 1 else 0.0
        dropout_delta_nse_mean[node_id] = (
            mean_delta_nse,
            mean_relative_residual,
            std_delta_nse,
            std_relative_residual
        )

    overall_event_means = []
    for event_item in event_matrices:
        event_reconstructed = _reconstruct_from_measurements(
            _selected_sensor_measurements(event_item['data'], selected_sensors, context),
            selected_sensors,
            Psi,
            r,
            context
        )
        overall_event_means.append(np.mean(_nse_values(event_reconstructed, event_item['data'], Node_ID)))
    overall_baseline_mean_nse = float(np.mean(overall_event_means)) if overall_event_means else np.nan
    return overall_baseline_mean_nse, dropout_delta_nse, dropout_delta_nse_mean


def Calculate_dropout_diagnostics(Node_ID, r, Psi, context, event_matrices, system_level_event_nse=False):
    ind_Psi = np.arange(r)
    _, _, pivoting = qr(Psi[:, ind_Psi].T, pivoting=True)
    selected_sensors = pivoting[:r]
    Psi_original = Psi[:, :r]
    diagnostics = []

    for drop_index in selected_sensors:
        remaining_sensors = [s for s in selected_sensors if s != drop_index]
        remaining_node_indices = [idx for idx in range(Psi_original.shape[0]) if idx != drop_index]
        Psi_red = Psi[remaining_sensors, :r]
        if system_level_event_nse:
            baseline_mean_nse, dropout_mean_nse, baseline_event_values, dropout_event_values = _eventwise_dropout_system_level_statistics(
                event_matrices, Node_ID, selected_sensors, drop_index, Psi, r, context
            )
            baseline_node_mean = None
            dropout_node_mean = None
        else:
            baseline_mean_nse, dropout_mean_nse, baseline_node_mean, dropout_node_mean = _eventwise_dropout_nse_statistics(
                event_matrices, Node_ID, selected_sensors, drop_index, Psi, r, context
            )
        delta_mean_nse = baseline_mean_nse - dropout_mean_nse

        current_row = Psi_original[drop_index, :]
        other_rows = Psi_original[remaining_sensors, :]
        if other_rows.shape[0] > 0:
            projection_matrix = other_rows.T @ np.linalg.pinv(other_rows @ other_rows.T) @ other_rows
            projection_of_row = projection_matrix @ current_row.T
            residual_vector = current_row.T - projection_of_row
            residual_norm = norm(residual_vector)
            row_norm = norm(current_row.T)
            relative_residual = residual_norm / row_norm if row_norm > 0 else 0.0
        else:
            relative_residual = 1.0

        singular_values = np.linalg.svd(Psi_red, compute_uv=False)
        sigma_min = float(np.min(singular_values)) if singular_values.size > 0 else 0.0
        sigma_max = float(np.max(singular_values)) if singular_values.size > 0 else 0.0
        condition_number = float(np.inf) if sigma_min == 0.0 else float(sigma_max / sigma_min)

        diagnostics.append({
            'Node_ID': Node_ID[drop_index],
            'Sensor_Index': int(drop_index),
            'r': int(r),
            'Baseline_Mean_NSE': float(baseline_mean_nse),
            'Dropout_Mean_NSE': float(dropout_mean_nse),
            'Delta_Mean_NSE': float(delta_mean_nse),
            'Relative_Projection_Residual': float(relative_residual),
            'Condition_Number': condition_number,
            'Sigma_Min': sigma_min,
            'Sigma_Max': sigma_max,
        })

        if event_matrices is not None and system_level_event_nse:
            diagnostics[-1]['Baseline_System_Level_Mean_NSE'] = float(baseline_mean_nse)
            diagnostics[-1]['Dropout_System_Level_Mean_NSE'] = float(dropout_mean_nse)
            diagnostics[-1]['Delta_System_Level_Mean_NSE'] = float(delta_mean_nse)
            diagnostics[-1]['Baseline_System_Level_NSE'] = baseline_event_values
            diagnostics[-1]['Dropout_System_Level_NSE'] = dropout_event_values

        if baseline_node_mean is not None and dropout_node_mean is not None:
            common_nodes = sorted(set(baseline_node_mean.keys()) & set(dropout_node_mean.keys()))
            diagnostics[-1]['Node_Level_Delta_Map'] = {
                node: float(baseline_node_mean[node] - dropout_node_mean[node]) for node in common_nodes
            }

    return diagnostics


def Calculate_MSE(data, Node_ID, r_values, Psi, context):
    mse_values = {}
    node_data = defaultdict(list)

    for r in r_values:
        ind_Psi = np.arange(r)
        _, _, pivoting = qr(Psi[:, ind_Psi].T, pivoting=True)
        selected_sensors = pivoting[:r]
        Psi_original = Psi[:, :r]

        for _, drop_index in enumerate(selected_sensors):
            remaining_sensors = [s for s in selected_sensors if s != drop_index]
            Psi_remaining_sensors = [s for s in range(Psi_original.shape[0]) if s != drop_index]
            Psi_original_red = Psi_original[Psi_remaining_sensors, :r]
            Psi_red = Psi[remaining_sensors, :r]
            pinv_red = np.linalg.pinv(Psi_red)

            y = data
            y_normalized = (y - _ctx(context, 'X_mean')) / _ctx(context, 'X_std_safe')
            x_hat_physical_d = (Psi_original_red @ pinv_red @ y_normalized[remaining_sensors]) * _ctx(context, 'X_std_safe')[Psi_remaining_sensors] + _ctx(context, 'X_mean')[Psi_remaining_sensors]
            x_hat_physical_d = _apply_physical_constraints(
                x_hat_physical_d,
                context,
                node_indices=Psi_remaining_sensors
            )
            y_r = y[Psi_remaining_sensors]
            avg_mse = np.mean((x_hat_physical_d - y_r) ** 2)
            node_data[Node_ID[drop_index]].append(avg_mse)

    for node_id, values in node_data.items():
        mse_values[node_id] = sum(values) / len(values)
    return mse_values
