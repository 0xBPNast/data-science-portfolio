"""
lpdm.py — 8-DOF Lumped Parameter Dynamic Model (LPDM)

Implements the planar multi-body gearbox model from Mohammed et al. (2015),
parameterised from Chaari et al. (2012). Provides system matrix construction
and the Newmark-Beta implicit time integration solver.

State vector order: [xp, yp, theta_m, theta_p, xg, yg, theta_b, theta_g]

References
----------
Mohammed, O.D. et al. (2015). Influence of dynamic modelling on the fault
    detection capability for helical gears.
Chaari, F. et al. (2012). Gearbox vibration signal amplitude and frequency
    modulation.
"""

import numpy as np


# ── System Matrices ────────────────────────────────────────────────────────────

def mass_matrix(m_1, m_2, i_11, i_12, i_21, i_22):
    """
    8x8 diagonal mass/inertia matrix.

    Parameters
    ----------
    m_1  : float — pinion mass [kg]
    m_2  : float — gear mass [kg]
    i_11 : float — motor moment of inertia [kg.m²]
    i_12 : float — pinion moment of inertia [kg.m²]
    i_21 : float — gear moment of inertia [kg.m²]
    i_22 : float — load moment of inertia [kg.m²]

    Returns
    -------
    M : array (8x8)
    """
    return np.array([
        [m_1, 0,    0,    0,    0,   0,    0,    0   ],
        [0,   m_1,  0,    0,    0,   0,    0,    0   ],
        [0,   0,    i_11, 0,    0,   0,    0,    0   ],
        [0,   0,    0,    i_12, 0,   0,    0,    0   ],
        [0,   0,    0,    0,    m_2, 0,    0,    0   ],
        [0,   0,    0,    0,    0,   m_2,  0,    0   ],
        [0,   0,    0,    0,    0,   0,    i_22, 0   ],
        [0,   0,    0,    0,    0,   0,    0,    i_21]
    ])


def k_static(k_x1, k_y1, k_x2, k_y2, k_th1, k_th2):
    """
    Static stiffness matrix.

    Includes translational bearing stiffness for pinion and gear,
    and torsional shaft stiffness coupling motor-pinion and load-gear.

    Parameters
    ----------
    k_x1, k_y1 : float — pinion translational stiffness [N/m]
    k_x2, k_y2 : float — gear translational stiffness [N/m]
    k_th1       : float — motor-pinion shaft torsional stiffness [N.m/rad]
    k_th2       : float — load-gear shaft torsional stiffness [N.m/rad]

    Returns
    -------
    K : array (8x8)
    """
    return np.array([
        [k_x1, 0,    0,      0,      0,    0,    0,      0     ],
        [0,    k_y1, 0,      0,      0,    0,    0,      0     ],
        [0,    0,    k_th1, -k_th1,  0,    0,    0,      0     ],
        [0,    0,   -k_th1,  k_th1,  0,    0,    0,      0     ],
        [0,    0,    0,      0,      k_x2, 0,    0,      0     ],
        [0,    0,    0,      0,      0,    k_y2, 0,      0     ],
        [0,    0,    0,      0,      0,    0,    k_th2, -k_th2 ],
        [0,    0,    0,      0,      0,    0,   -k_th2,  k_th2 ]
    ])


def c_matrix(c_m, c_k, m_mat, k_mat):
    """
    Proportional (Rayleigh) damping matrix.

    C = c_m * M + c_k * K(t)

    Recomputed at each time step since K(t) varies with TVMS.

    Parameters
    ----------
    c_m   : float — mass-proportional damping coefficient
    c_k   : float — stiffness-proportional damping coefficient
    m_mat : array — mass matrix (8x8)
    k_mat : array — current stiffness matrix (8x8)

    Returns
    -------
    C : array (8x8)
    """
    return c_m * m_mat + c_k * k_mat


def s_matrix(alpha, r_pinion, r_gear):
    """
    Geometry matrix relating meshing displacement to DOF forces.

    Derived from tooth contact geometry at pressure angle alpha.

    Parameters
    ----------
    alpha    : float — pressure angle [rad]
    r_pinion : float — pinion pitch radius [m]
    r_gear   : float — gear pitch radius [m]

    Returns
    -------
    S : array (8x8)
    """
    s3  = np.sin(alpha) ** 2
    s4  = np.cos(alpha) ** 2
    s5  = np.sin(alpha) * np.cos(alpha)
    s6  = r_pinion * np.cos(alpha)
    s7  = r_pinion * np.sin(alpha)
    s8  = r_gear * np.cos(alpha)
    s9  = r_gear * np.sin(alpha)
    s10 = r_pinion ** 2
    s11 = r_gear ** 2
    s12 = r_pinion * r_gear

    return np.array([
        [ s3,  s5, 0,  s7, -s3, -s5, 0,  s9 ],
        [ s5,  s4, 0,  s6, -s5, -s4, 0,  s8 ],
        [  0,   0, 0,   0,   0,   0, 0,   0 ],
        [ s7,  s6, 0, s10, -s7, -s6, 0, s12 ],
        [-s3, -s5, 0, -s7,  s3,  s5, 0, -s9 ],
        [-s5, -s4, 0, -s6,  s5,  s4, 0, -s8 ],
        [  0,   0, 0,   0,   0,   0, 0,   0 ],
        [ s9,  s8, 0, s12, -s9, -s8, 0, s11 ]
    ])


# ── Default System Parameters (Chaari et al., 2012) ───────────────────────────

DEFAULT_PARAMS = {
    'm_1'   : 0.6,       # Pinion mass [kg]
    'm_2'   : 1.5,       # Gear mass [kg]
    'i_11'  : 0.0043,    # Motor moment of inertia [kg.m²]
    'i_12'  : 0.00027,   # Pinion moment of inertia [kg.m²]
    'i_21'  : 0.0027,    # Gear moment of inertia [kg.m²]
    'i_22'  : 0.0045,    # Load moment of inertia [kg.m²]
    'k_x1'  : 1e8,       # Pinion x-bearing stiffness [N/m]
    'k_y1'  : 1e8,       # Pinion y-bearing stiffness [N/m]
    'k_x2'  : 1e8,       # Gear x-bearing stiffness [N/m]
    'k_y2'  : 1e8,       # Gear y-bearing stiffness [N/m]
    'k_th1' : 1e5,       # Motor-pinion torsional stiffness [N.m/rad]
    'k_th2' : 1e5,       # Load-gear torsional stiffness [N.m/rad]
    'alpha' : 20.0,      # Pressure angle [degrees]
    'r_p'   : 0.02819,   # Pinion pitch radius [m]
    'r_g'   : 0.05638,   # Gear pitch radius [m]
    'c_m'   : 0.05,      # Mass-proportional damping coefficient
    'c_k'   : 1e-6,      # Stiffness-proportional damping coefficient
    'Nt_p'  : 20,        # Pinion teeth
    'Nt_g'  : 40,        # Gear teeth
}


def build_system_matrices(params=None):
    """
    Construct M, K_static, and S matrices from parameter dict.

    Parameters
    ----------
    params : dict — system parameters (uses DEFAULT_PARAMS if None)

    Returns
    -------
    M, K, S : arrays (8x8 each)
    """
    p = params or DEFAULT_PARAMS
    M = mass_matrix(p['m_1'], p['m_2'], p['i_11'], p['i_12'], p['i_21'], p['i_22'])
    K = k_static(p['k_x1'], p['k_y1'], p['k_x2'], p['k_y2'], p['k_th1'], p['k_th2'])
    S = s_matrix(np.radians(p['alpha']), p['r_p'], p['r_g'])
    return M, K, S


# ── Newmark-Beta Solver ────────────────────────────────────────────────────────

def newmark_solver(n_steps, dt, M, K_static, S_matrix, c_m, c_k,
                   beta, gamma, initial_conditions,
                   force_func, stiffness_func,
                   theta_cycle, contact_ratio):
    """
    Newmark-Beta implicit time integration solver.

    Solves M*q'' + C*q' + K(t)*q = F(t) iteratively.
    The stiffness matrix K(t) is updated at every step via the TVMS.
    Unconditionally stable with beta=0.25, gamma=0.5.

    Parameters
    ----------
    n_steps            : int      — number of time steps
    dt                 : float    — time step [s]
    M                  : array    — mass matrix (8x8)
    K_static           : array    — static stiffness matrix (8x8)
    S_matrix           : array    — geometry matrix (8x8)
    c_m, c_k           : float    — proportional damping coefficients
    beta, gamma        : float    — Newmark parameters (0.25, 0.5 recommended)
    initial_conditions : dict     — {'u0', 'v0', 'a0'} initial state vectors
    force_func         : callable — F(omega_r, time_i) → array (8,)
    stiffness_func     : callable — k_gm(theta, theta_cycle, contact_ratio) → float
    theta_cycle        : float    — angular displacement per mesh cycle [rad]
    contact_ratio      : float    — gear contact ratio

    Returns
    -------
    u          : array (n_steps x 8) — displacement
    v          : array (n_steps x 8) — velocity
    a          : array (n_steps x 8) — acceleration
    F_store    : array (n_steps x 8) — force vectors
    k_gm_store : array (n_steps,)    — TVMS values
    """
    dof        = M.shape[0]
    u          = np.zeros((n_steps, dof))
    v          = np.zeros((n_steps, dof))
    a          = np.zeros((n_steps, dof))
    F_store    = np.zeros((n_steps, dof))
    k_gm_store = np.zeros(n_steps)

    # Initial conditions
    u[0] = initial_conditions.get('u0', np.zeros(dof))
    v[0] = initial_conditions.get('v0', np.zeros(dof))
    a[0] = initial_conditions.get('a0', np.zeros(dof))

    k_gm_store[0] = stiffness_func(u[0][3], theta_cycle, contact_ratio)
    K_full        = K_static + k_gm_store[0] * S_matrix
    C_mat         = c_matrix(c_m, c_k, M, K_full)
    F_store[0, :] = force_func(v[0, 2], time_i=0)
    a[0, :]       = np.linalg.solve(M, F_store[0, :] - C_mat @ v[0, :] - K_full @ u[0, :])

    for i in range(1, n_steps):
        k_gm_value    = stiffness_func(u[i-1, 3], theta_cycle, contact_ratio)
        k_gm_store[i] = k_gm_value
        K_full        = K_static + k_gm_value * S_matrix
        C_mat         = c_matrix(c_m, c_k, M, K_full)

        K_eff = (K_full
                 + (gamma / (beta * dt)) * C_mat
                 + (1 / (beta * dt**2)) * M)

        F_eff = (F_store[i-1]
                 + M @ ((1 / (beta * dt**2)) * u[i-1]
                        + (1 / (beta * dt)) * v[i-1]
                        + (1 / (2*beta) - 1) * a[i-1])
                 + C_mat @ ((gamma / (beta * dt)) * u[i-1]
                            + (gamma / beta - 1) * v[i-1]
                            + dt * (gamma / (2*beta) - 1) * a[i-1]))

        u[i] = np.linalg.solve(K_eff, F_eff)
        a[i] = ((1 / (beta * dt**2)) * (u[i] - u[i-1])
                - (1 / (beta * dt)) * v[i-1]
                - (1 / (2*beta) - 1) * a[i-1])
        v[i] = v[i-1] + dt * ((1 - gamma) * a[i-1] + gamma * a[i])

        F_store[i, :] = force_func(v[i, 2], time_i=i * dt)

    return u, v, a, F_store, k_gm_store
