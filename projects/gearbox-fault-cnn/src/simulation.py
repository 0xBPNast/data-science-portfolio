"""
simulation.py — High-Level Simulation Runner

Provides a single entry point for running all six LPDM dataset configurations
and saving outputs to HDF5. Combines lpdm.py and tvms.py into a clean
end-to-end simulation pipeline.

Usage
-----
    from src.simulation import run_all_simulations, save_to_hdf5

    results = run_all_simulations()
    save_to_hdf5(results, 'data/gearbox_data.h5')
"""

import numpy as np
import h5py

from .lpdm import build_system_matrices, newmark_solver, DEFAULT_PARAMS
from .tvms import get_stiffness_func, get_force_func, DEFAULT_MOTOR


# ── Dataset Configurations ─────────────────────────────────────────────────────

DATASET_CONFIGS = [
    {'id': '1_C_H',  'load_type': 'constant', 'fault_type': None},
    {'id': '2_C_CT', 'load_type': 'constant', 'fault_type': 'cracked_tooth'},
    {'id': '3_C_WT', 'load_type': 'constant', 'fault_type': 'worn_teeth'},
    {'id': '4_V_H',  'load_type': 'varying',  'fault_type': None},
    {'id': '5_V_CT', 'load_type': 'varying',  'fault_type': 'cracked_tooth'},
    {'id': '6_V_WT', 'load_type': 'varying',  'fault_type': 'worn_teeth'},
]


# ── Simulation Runner ──────────────────────────────────────────────────────────

def run_simulation(config, dt=5e-6, total_time=5.0,
                   wear_factor=0.9, params=None, motor=None):
    """
    Run a single LPDM simulation for a given dataset configuration.

    Parameters
    ----------
    config     : dict  — dataset config with keys 'id', 'load_type', 'fault_type'
    dt         : float — time step [s] (default 5e-6 → Fs = 200 kHz)
    total_time : float — simulation duration [s]
    wear_factor: float — stiffness reduction for faulted teeth
                         (0.9 = 10% reduction, 0.6 = 40% reduction)
    params     : dict  — system parameters (uses DEFAULT_PARAMS if None)
    motor      : dict  — motor parameters (uses DEFAULT_MOTOR if None)

    Returns
    -------
    dict with keys:
        'displacement' : array (n_steps x 8)
        'velocity'     : array (n_steps x 8)
        'acceleration' : array (n_steps x 8)
        'force'        : array (n_steps x 8)
        'stiffness'    : array (n_steps,)
    """
    p = params or DEFAULT_PARAMS
    m = motor  or DEFAULT_MOTOR

    n_steps    = int(total_time / dt)
    dof        = 8
    beta, gamma = 0.25, 0.5

    M, K_static, S = build_system_matrices(p)

    theta_cycle   = 2 * np.pi / p['Nt_p']
    contact_ratio = 1.6
    omega_s       = m['omega_s']

    initial_conditions = {
        'u0': np.zeros(dof),
        'v0': np.zeros(dof),
        'a0': np.zeros(dof)
    }

    dynamic_load = config['load_type'] == 'varying'

    force_func     = get_force_func(omega_s=omega_s, dynamic=dynamic_load,
                                    t_b=m['t_b'], g_b=m['g_b'],
                                    c_a1=m['c_a1'], c_a2=m['c_a2'])
    stiffness_func = get_stiffness_func(fault_type=config['fault_type'],
                                        wear_factor=wear_factor)

    u, v, a, F_store, k_gm_store = newmark_solver(
        n_steps=n_steps, dt=dt,
        M=M, K_static=K_static, S_matrix=S,
        c_m=p['c_m'], c_k=p['c_k'],
        beta=beta, gamma=gamma,
        initial_conditions=initial_conditions,
        force_func=force_func,
        stiffness_func=stiffness_func,
        theta_cycle=theta_cycle,
        contact_ratio=contact_ratio
    )

    return {
        'displacement': u,
        'velocity':     v,
        'acceleration': a,
        'force':        F_store,
        'stiffness':    k_gm_store
    }


def run_all_simulations(configs=None, dt=5e-6, total_time=5.0,
                        wear_factor=0.9, params=None, motor=None,
                        verbose=True):
    """
    Run all six dataset simulations and return results dict.

    Parameters
    ----------
    configs    : list  — dataset configs (uses DATASET_CONFIGS if None)
    dt         : float — time step [s]
    total_time : float — simulation duration [s]
    wear_factor: float — fault stiffness reduction multiplier
    params     : dict  — system parameters
    motor      : dict  — motor parameters
    verbose    : bool  — print progress

    Returns
    -------
    results    : dict  — keyed by dataset ID, each value is simulation output dict
    time_array : array — time vector (n_steps,)
    """
    configs    = configs or DATASET_CONFIGS
    n_steps    = int(total_time / dt)
    time_array = np.linspace(0, total_time, n_steps)
    results    = {}

    for config in configs:
        if verbose:
            print(f"Running: {config['id']}  "
                  f"(load={config['load_type']}, fault={config['fault_type']})...")
        results[config['id']] = run_simulation(
            config, dt=dt, total_time=total_time,
            wear_factor=wear_factor, params=params, motor=motor
        )
        if verbose:
            print(f"  Complete.")

    if verbose:
        print(f"\nAll {len(configs)} simulations complete.")

    return results, time_array


# ── HDF5 I/O ──────────────────────────────────────────────────────────────────

def save_to_hdf5(results, time_array, file_path, group_prefix='',
                 compression='gzip', compression_opts=6):
    """
    Save simulation results to an HDF5 file.

    Stores pinion y-acceleration, pinion angular velocity, and motor
    angular velocity for each dataset under its ID as a group.

    Parameters
    ----------
    results          : dict   — simulation results from run_all_simulations
    time_array       : array  — time vector
    file_path        : str    — output HDF5 file path
    group_prefix     : str    — optional prefix for group names (e.g. 'unseen/')
    compression      : str    — HDF5 compression algorithm
    compression_opts : int    — compression level (1–9)
    """
    mode = 'a' if group_prefix else 'w'   # Append for unseen data, write for training

    with h5py.File(file_path, mode) as hdf:
        hdf.create_dataset(f'{group_prefix}time',
                           data=time_array,
                           compression=compression,
                           compression_opts=compression_opts)

        for dataset_id, data in results.items():
            grp = hdf.create_group(f'{group_prefix}{dataset_id}')
            grp.create_dataset('Pinion_Y_Acceleration',
                               data=data['acceleration'][:, 1],
                               compression=compression,
                               compression_opts=compression_opts)
            grp.create_dataset('Pinion_Angular_Velocity',
                               data=data['velocity'][:, 3],
                               compression=compression,
                               compression_opts=compression_opts)
            grp.create_dataset('Motor_Angular_Velocity',
                               data=data['velocity'][:, 2],
                               compression=compression,
                               compression_opts=compression_opts)

    print(f"Saved {len(results)} datasets to {file_path}")


def load_from_hdf5(file_path, dataset_ids, group_prefix=''):
    """
    Load vibration and rotation data from HDF5 file.

    Parameters
    ----------
    file_path   : str  — path to HDF5 file
    dataset_ids : list — list of dataset ID strings to load
    group_prefix: str  — optional group prefix (e.g. 'unseen/')

    Returns
    -------
    vibration_data : list — pinion y-acceleration arrays
    rotation_data  : list — pinion angular velocity arrays
    """
    vibration_data, rotation_data = [], []

    with h5py.File(file_path, 'r') as hdf:
        for ds_id in dataset_ids:
            key = f'{group_prefix}{ds_id}'
            vibration_data.append(hdf[key]['Pinion_Y_Acceleration'][:])
            rotation_data.append(hdf[key]['Pinion_Angular_Velocity'][:])

    return vibration_data, rotation_data
