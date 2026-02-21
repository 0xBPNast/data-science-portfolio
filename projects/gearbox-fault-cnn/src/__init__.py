"""
src — Gearbox Fault Classification

Source modules for the gearbox fault CNN portfolio project.

Modules
-------
lpdm          : 8-DOF Lumped Parameter Dynamic Model and Newmark-Beta solver
tvms          : Time-Varying Meshing Stiffness and motor dynamics
preprocessing : Signal preprocessing pipeline (downsample, COT, segment, augment)
cnn           : 1D-CNN architecture, training, and normalisation utilities
simulation    : High-level simulation runner and HDF5 I/O
"""

from .lpdm          import mass_matrix, k_static, c_matrix, s_matrix, newmark_solver, build_system_matrices, DEFAULT_PARAMS
from .tvms          import k_gm_with_fault, motor_torque, force_vector, get_stiffness_func, get_force_func, DEFAULT_MOTOR
from .preprocessing import cot, generate_tachometer, downsample_signal, segment_signal, augment_with_noise, to_frequency_domain, preprocess_vibration_data, process_and_segment
from .cnn           import build_1d_cnn, normalize_and_standardize, normalize_for_inference, train_and_evaluate_1d_cnn
from .simulation    import run_simulation, run_all_simulations, save_to_hdf5, load_from_hdf5, DATASET_CONFIGS
