import logging
import os
import re
import sys
import traceback

import numpy as np
import pandas as pd
import polars as pl
from matplotlib import rcParams
from scipy.linalg import svd

rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial']
rcParams['mathtext.fontset'] = 'dejavusans'
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

CURRENT_DIR = os.path.dirname(__file__)
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

MODEL_DIRECTORY = os.path.join(CURRENT_DIR, 'saved_models')

from DSS_menu_handlers import dispatch_menu_choice, print_menu

SIMULATION_PATH = os.environ.get('DSS_SWMM_MODEL_PATH')
SIMULATION_PATH = os.path.abspath(os.path.expanduser(SIMULATION_PATH)) if SIMULATION_PATH else None
TARGET_NODE_IDS = ['184', '148', 'OF-02', '84', '34']
OUTPUT_CSV_BASE_PATH = os.path.join(CURRENT_DIR, '..', 'SWMM_Model', 'csv_files', 'node_flow_')
RAINFALL_EVENTS = []
RAINFALL_EVENTS_SYNTHETIC = ['200-year', '100-year', '050-year', '025-year', '010-year', '005-year']
RAINFALL_EVENTS_TRAINING = ['8-26', '9-3', '5-21', '6-15', '8-15']
RAINFALL_EVENTS_PREDICTION = ['9-5', '9-17', '6-2', '6-11', '6-28']
IMPERVIOUSNESS = ['05', '15', '25', '35', '45']
MINIMUM_RATE = ['0.5', '1.0', '1.5', '2.0', '2.5']
CSV_DIRECTORY = os.path.join(CURRENT_DIR, '..', 'SWMM_Model', 'csv_files')
TESTING_FLOWRATES_DIRECTORY = os.path.join(CURRENT_DIR, '..', 'Testing', 'Flowrates_data')
TRAINING_FLOWRATES_PATH = os.path.join(CURRENT_DIR, '..', 'Training', 'Flowrates_data', 'Training_flowrates_dataset.csv')


def load_training_dataset(training_file_path):
    if not os.path.exists(training_file_path):
        raise FileNotFoundError(
            "Training flow-rate dataset was not found.\n"
            f"Expected: {training_file_path}\n"
            "Keep the repository Training/Flowrates_data dataset in place for reproducible analysis."
        )

    loaded = pl.read_csv(training_file_path, low_memory=False)
    node_id_raw = loaded[:, 0]
    node_id = [re.sub(r'(OF)(\d+)', r'\1-\2', str(raw_id)) for raw_id in node_id_raw]
    data_matrix = loaded[:, 1:].to_numpy()
    return node_id_raw, node_id, data_matrix


Node_ID_raw, Node_ID, X = load_training_dataset(TRAINING_FLOWRATES_PATH)

X_max_limits = np.max(X, axis=1, keepdims=True)
X_mean_zscore = np.mean(X, axis=1, keepdims=True)
X_std = np.std(X, axis=1, keepdims=True)
X_std_safe_zscore = np.where(X_std == 0, 1.0, X_std)
X_global_minmax_min = float(np.min(X))
X_global_minmax_max = float(np.max(X))
X_global_minmax_scale = X_global_minmax_max - X_global_minmax_min
X_global_minmax_scale = X_global_minmax_scale if X_global_minmax_scale > 0 else 1.0
X_mean_global_minmax = np.full_like(X_mean_zscore, X_global_minmax_min, dtype=float)
X_std_safe_global_minmax = np.full_like(X_mean_zscore, X_global_minmax_scale, dtype=float)

DEFAULT_NORMALIZATION_MODE = 'global_minmax'


def build_normalization_config(mode=DEFAULT_NORMALIZATION_MODE):
    mode_key = str(mode).strip().lower()
    if mode_key in {'2', 'global_minmax', 'global minmax', 'global-minmax', 'global_max', 'global max', 'global-max', 'max'}:
        return {
            'mode': 'global_minmax',
            'label': 'Global-MinMax scale',
            'X_mean': X_mean_global_minmax,
            'X_std_safe': X_std_safe_global_minmax,
            'X_scale_scalar': X_global_minmax_scale,
            'svd_suffix': 'global_minmax',
        }

    return {
        'mode': 'zscore',
        'label': 'Z-score',
        'X_mean': X_mean_zscore,
        'X_std_safe': X_std_safe_zscore,
        'X_scale_scalar': None,
        'svd_suffix': 'zscore',
    }


def choose_normalization_mode():
    print("\nSelect normalization mode:")
    print("1. Z-score")
    print("2. Global-MinMax scale (default)")
    choice = input("Enter normalization mode (1-2, default: 2): ").strip().lower()
    if choice in {'2', 'global_minmax', 'global minmax', 'global-minmax', 'global_max', 'global max', 'global-max', 'max'}:
        return 'global_minmax'
    if choice in {'', '1', 'zscore', 'z-score', 'z score'}:
        return DEFAULT_NORMALIZATION_MODE if choice == '' else 'zscore'
    return DEFAULT_NORMALIZATION_MODE


def load_testing_event_matrices():
    event_files = [
        'Flowrates_6-2_2024.csv',
        'Flowrates_6-11_2024.csv',
        'Flowrates_6-28_2024.csv',
        'Flowrates_9-5_2019.csv',
        'Flowrates_9-17_2019.csv',
        'Flowrates_200-year.csv',
    ]
    event_matrices = []
    for event_file in event_files:
        event_path = os.path.join(TESTING_FLOWRATES_DIRECTORY, event_file)
        event_df = pd.read_csv(event_path)
        event_data = event_df.iloc[:, 1:].to_numpy()
        event_matrices.append({
            'event_name': os.path.splitext(event_file)[0],
            'data': event_data,
            'columns': event_df.columns[1:].tolist(),
        })
    return event_matrices


def get_svd_save_path(normalization_mode=DEFAULT_NORMALIZATION_MODE):
    norm_config = build_normalization_config(normalization_mode)
    return os.path.join(MODEL_DIRECTORY, f"svd_decomposition_{norm_config['svd_suffix']}.npz")


def ensure_model_directory():
    os.makedirs(MODEL_DIRECTORY, exist_ok=True)


def migrate_legacy_svd_file(normalization_mode=DEFAULT_NORMALIZATION_MODE):
    legacy_path = os.path.join(CURRENT_DIR, 'svd_decomposition_standardized.npz')
    target_path = get_svd_save_path(normalization_mode)
    legacy_mode_path = os.path.join(MODEL_DIRECTORY, 'svd_decomposition_global_max.npz')

    if os.path.exists(target_path):
        return target_path

    if os.path.exists(legacy_mode_path):
        os.replace(legacy_mode_path, target_path)
        print(f"Renamed legacy normalization SVD file to: {target_path}")
        return target_path

    if not os.path.exists(legacy_path):
        return target_path

    ensure_model_directory()
    os.replace(legacy_path, target_path)
    print(f"Moved legacy SVD file to dedicated model folder: {target_path}")
    return target_path


def load_default_svd_decomposition(normalization_mode=DEFAULT_NORMALIZATION_MODE):
    ensure_model_directory()
    norm_config = build_normalization_config(normalization_mode)
    svd_save_path = migrate_legacy_svd_file(normalization_mode)

    if os.path.exists(svd_save_path):
        svd_data = np.load(svd_save_path)
        Psi = svd_data['Psi']
        S = svd_data['S']
        V = svd_data['V']
        print(f"SVD decomposition results ({svd_save_path}) loaded.")
        return Psi, S, V

    print(f"Computing default SVD basis using {norm_config['label']} normalization...")
    X_normalized = (X - norm_config['X_mean']) / norm_config['X_std_safe']
    Psi, S, V = svd(X_normalized, full_matrices=False)
    np.savez(svd_save_path, Psi=Psi, S=S, V=V)
    print("SVD decomposition results computed and saved.")
    return Psi, S, V


def build_analysis_context(normalization_config=None):
    normalization_config = normalization_config or build_normalization_config()
    return {
        'X_mean': normalization_config['X_mean'],
        'X_std_safe': normalization_config['X_std_safe'],
        'X_max_limits': X_max_limits,
        'Node_ID': Node_ID,
        'normalization_mode': normalization_config['mode'],
        'normalization_label': normalization_config['label'],
        'X_scale_scalar': normalization_config['X_scale_scalar'],
    }


def build_swmm_config():
    return {
        'CURRENT_DIR': CURRENT_DIR,
        'SIMULATION_PATH': SIMULATION_PATH,
        'TARGET_NODE_IDS': TARGET_NODE_IDS,
        'OUTPUT_CSV_BASE_PATH': OUTPUT_CSV_BASE_PATH,
        'RAINFALL_EVENTS': RAINFALL_EVENTS,
        'RAINFALL_EVENTS_SYNTHETIC': RAINFALL_EVENTS_SYNTHETIC,
        'RAINFALL_EVENTS_TRAINING': RAINFALL_EVENTS_TRAINING,
        'RAINFALL_EVENTS_PREDICTION': RAINFALL_EVENTS_PREDICTION,
        'IMPERVIOUSNESS': IMPERVIOUSNESS,
        'MINIMUM_RATE': MINIMUM_RATE,
        'CSV_DIRECTORY': CSV_DIRECTORY,
    }


def build_state(Psi, S, V, normalization_config=None):
    normalization_config = normalization_config or build_normalization_config()
    return {
        'CURRENT_DIR': CURRENT_DIR,
        'MODEL_DIRECTORY': MODEL_DIRECTORY,
        'Node_ID': Node_ID,
        'Node_ID_raw': Node_ID_raw,
        'X': X,
        'X_mean': normalization_config['X_mean'],
        'X_std_safe': normalization_config['X_std_safe'],
        'X_max_limits': X_max_limits,
        'normalization_mode': normalization_config['mode'],
        'normalization_label': normalization_config['label'],
        'X_scale_scalar': normalization_config['X_scale_scalar'],
        'Psi': Psi,
        'S': S,
        'V': V,
        'testing_event_matrices': load_testing_event_matrices(),
        'analysis_context': build_analysis_context(normalization_config),
        'swmm_config': build_swmm_config(),
    }


def main():
    try:
        normalization_mode = choose_normalization_mode()
        normalization_config = build_normalization_config(normalization_mode)
        print(f"Using normalization mode: {normalization_config['label']}")
        Psi, S, V = load_default_svd_decomposition(normalization_mode)
        state = build_state(Psi, S, V, normalization_config)

        while True:
            print_menu()
            choice = input("Enter option (1-23): ")
            if not dispatch_menu_choice(choice, state):
                break
    except Exception as e:
        tb = traceback.format_exc()
        logging.error(f"An error occurred: {e}\nStack trace: \n{tb}")


if __name__ == "__main__":
    main()
