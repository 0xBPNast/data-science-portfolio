"""
tvms.py — Time-Varying Meshing Stiffness (TVMS) & Motor Dynamics

Implements the rectangular step-function TVMS model from Chaari et al. (2012)
with optional fault introduction (tooth crack, worn teeth), and the squirrel
cage induction motor torque-speed model used to excite the gearbox.

References
----------
Chaari, F. et al. (2012). Gearbox vibration signal amplitude and frequency
    modulation.
"""

import numpy as np


# ── Time-Varying Meshing Stiffness ─────────────────────────────────────────────

def k_gm_with_fault(theta_current, theta_cycle, c,
                    k_gm_min, k_gm_max,
                    fault_tooth=None, worn_teeth_range=None, wear_factor=1.0):
    """
    Time-Varying Meshing Stiffness with optional fault introduction.

    Implements the rectangular step TVMS from Chaari et al. (2012).
    Alternates between k_gm_min (single tooth pair in contact) and
    k_gm_max (two tooth pairs in contact) based on angular position
    and contact ratio.

    Faults are introduced by locally reducing the meshing stiffness:
    - Tooth crack : single tooth, localised stiffness reduction
    - Worn teeth  : distributed reduction over a range of teeth

    Parameters
    ----------
    theta_current    : float — current pinion angular displacement [rad]
    theta_cycle      : float — angular displacement per mesh cycle [rad]
                               = 2*pi / number_of_pinion_teeth
    c                : float — gear contact ratio
    k_gm_min         : float — single tooth pair stiffness [N/m]
    k_gm_max         : float — double tooth pair stiffness [N/m]
    fault_tooth      : int   — index of cracked tooth (None = healthy)
    worn_teeth_range : range — indices of worn teeth (None = healthy)
    wear_factor      : float — stiffness multiplier for faulted teeth
                               e.g. 0.9 = 10% reduction, 0.6 = 40% reduction

    Returns
    -------
    k_gm : float — gear meshing stiffness at current position [N/m]
    """
    # Position within current gear rotation
    n_rotations = np.floor(theta_current / (2 * np.pi))
    theta_rel   = theta_current - (2 * np.pi * n_rotations)
    n_cycles    = np.floor(theta_rel / theta_cycle)
    theta_rel_c = theta_rel - (n_cycles * theta_cycle)

    # Base TVMS step function
    if theta_rel_c < (theta_cycle * (2 - c)):
        k_gm = k_gm_min
    else:
        k_gm = k_gm_max

    # Tooth crack: localised stiffness reduction at fault tooth index
    if fault_tooth is not None and n_cycles == (fault_tooth - 1):
        k_gm *= wear_factor

    # Worn teeth: distributed stiffness reduction over affected range
    if worn_teeth_range is not None and n_cycles in worn_teeth_range:
        k_gm *= wear_factor

    return k_gm


def get_stiffness_func(fault_type=None, fault_tooth=5, worn_teeth_range=None,
                       wear_factor=0.9, k_gm_min=0.81e8, k_gm_max=2.1e8):
    """
    Factory function returning a configured stiffness callable.

    Returns a lambda compatible with the newmark_solver stiffness_func
    interface: f(theta, theta_cycle, contact_ratio) -> float.

    Parameters
    ----------
    fault_type       : str   — 'cracked_tooth', 'worn_teeth', or None
    fault_tooth      : int   — tooth index for crack fault (default 5)
    worn_teeth_range : range — tooth indices for wear fault (default range(3,6))
    wear_factor      : float — stiffness multiplier (default 0.9 = 10% reduction)
    k_gm_min         : float — minimum meshing stiffness [N/m]
    k_gm_max         : float — maximum meshing stiffness [N/m]

    Returns
    -------
    stiffness_func : callable
    """
    if worn_teeth_range is None:
        worn_teeth_range = range(3, 6)

    if fault_type == 'cracked_tooth':
        _fault_tooth      = fault_tooth
        _worn_teeth_range = None
    elif fault_type == 'worn_teeth':
        _fault_tooth      = None
        _worn_teeth_range = worn_teeth_range
    else:
        _fault_tooth      = None
        _worn_teeth_range = None

    return lambda theta, theta_cycle, contact_ratio: k_gm_with_fault(
        theta, theta_cycle, contact_ratio,
        k_gm_min=k_gm_min, k_gm_max=k_gm_max,
        fault_tooth=_fault_tooth,
        worn_teeth_range=_worn_teeth_range,
        wear_factor=wear_factor
    )


# ── Motor Dynamics ─────────────────────────────────────────────────────────────

def motor_torque(omega_r, omega_s, t_b, g_b, c_a1, c_a2, t_start=27.0):
    """
    Squirrel cage induction motor torque-speed relationship.

    Parameters
    ----------
    omega_r : float — current rotor angular velocity [rad/s]
    omega_s : float — synchronous speed [rad/s]
    t_b     : float — breakdown torque [Nm]
    g_b     : float — slip at breakdown torque
    c_a1    : float — motor empirical constant 1
    c_a2    : float — motor empirical constant 2
    t_start : float — startup torque at zero speed [Nm]

    Returns
    -------
    T_m : float — motor output torque [Nm]
    """
    if np.abs(omega_r) < 1e-4:
        return t_start
    g_n = 1 - (omega_r / omega_s)
    return t_b / (1 + (g_b - g_n) ** 2 * ((c_a1 / g_n) - c_a2 * g_n ** 2))


def force_vector(omega_r, omega_s, t_b, g_b, c_a1, c_a2, time_i,
                 dynamic=True, t_start=27.0, t_load_override=None):
    """
    External force/torque vector F(t) for the equations of motion.

    Assembles the 8-element force vector with motor torque at DOF 2
    (theta_m) and load torque at DOF 6 (theta_b).

    Parameters
    ----------
    omega_r          : float — current motor angular velocity [rad/s]
    omega_s          : float — synchronous speed [rad/s]
    t_b              : float — motor breakdown torque [Nm]
    g_b              : float — slip at breakdown
    c_a1             : float — motor constant 1
    c_a2             : float — motor constant 2
    time_i           : float — current simulation time [s]
    dynamic          : bool  — True = sinusoidal varying load, False = constant
    t_start          : float — startup torque [Nm]
    t_load_override  : float — override load torque directly (ignores dynamic flag)

    Returns
    -------
    f_vec : array (8,) — external force/torque vector
    """
    if t_load_override is not None:
        t_l = t_load_override
    elif dynamic:
        t_l = 10 * np.sin(2 * np.pi * time_i) + 30   # Varying load [Nm]
    else:
        t_l = 20                                        # Constant load [Nm]

    t_m = motor_torque(omega_r, omega_s, t_b, g_b, c_a1, c_a2, t_start)

    f_vec    = np.zeros(8)
    f_vec[2] = t_m
    f_vec[6] = t_l
    return f_vec


def get_force_func(omega_s, dynamic=True, t_load_override=None,
                   t_b=32.0, g_b=0.315, c_a1=1.711, c_a2=1.316):
    """
    Factory function returning a configured force callable.

    Returns a lambda compatible with the newmark_solver force_func
    interface: f(omega_r, time_i) -> array (8,).

    Parameters
    ----------
    omega_s          : float — synchronous speed [rad/s]
    dynamic          : bool  — True = varying load
    t_load_override  : float — fixed load torque override [Nm]
    t_b              : float — motor breakdown torque [Nm]
    g_b, c_a1, c_a2  : float — motor constants

    Returns
    -------
    force_func : callable
    """
    return lambda omega_r, time_i: force_vector(
        omega_r, omega_s=omega_s, t_b=t_b, g_b=g_b,
        c_a1=c_a1, c_a2=c_a2, time_i=time_i,
        dynamic=dynamic, t_load_override=t_load_override
    )


# ── Default Motor Parameters (Chaari et al., 2012) ────────────────────────────

DEFAULT_MOTOR = {
    't_b'    : 32.0,    # Breakdown torque [Nm]
    'g_b'    : 0.315,   # Slip at breakdown
    'c_a1'   : 1.711,   # Motor constant 1
    'c_a2'   : 1.316,   # Motor constant 2
    'omega_s': 1500 * (2 * np.pi / 60),  # Synchronous speed [rad/s]
}
