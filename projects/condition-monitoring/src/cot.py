import numpy as np


def cot(tach, Fs_tach, ppr, trigger, vibration, Fs_vibration, orders):
    """
    Computed Order Tracking (COT).

    Resamples a vibration signal from the time domain to the angular domain
    by interpolating a fixed number of samples per shaft revolution.
    Eliminates frequency smearing caused by varying shaft speed, enabling
    reliable order spectrum analysis.

    Parameters
    ----------
    tach          : array — tachometer signal (1D)
    Fs_tach       : float — tachometer sampling frequency [Hz]
    ppr           : int   — tachometer pulses per revolution
    trigger       : float — tachometer pulse trigger value
    vibration     : array — vibration signal (1D)
    Fs_vibration  : float — vibration sampling frequency [Hz]
    orders        : int   — resampled points per revolution

    Returns
    -------
    t_cot   : array — resampled time array
    sig_cot : array — order-tracked vibration signal
    """
    # Tachometer time array
    N_tach  = np.size(tach)
    tf_tach = N_tach / Fs_tach
    t_tach  = np.linspace(0, tf_tach, N_tach)

    # Locate tachometer pulse indices
    pulse_ind = np.where(tach == trigger)[0]

    # Extract every PPRth pulse (one per revolution)
    N_windows = int(len(pulse_ind) / ppr)
    ppr_ind   = np.array([pulse_ind[ppr * i - 1] for i in range(1, N_windows + 1)])
    ppr_ind   = ppr_ind[ppr_ind != 0].astype(int)

    # Revolution boundary times from tachometer
    t_tach_ppr = t_tach[ppr_ind]

    # Vibration signal time array
    N_sig  = np.size(vibration)
    tf_sig = N_sig / Fs_vibration
    t_sig  = np.linspace(0, tf_sig, N_sig)

    # Map tachometer revolution boundaries to vibration signal indices
    # (vectorised np.argmin replaces slow linear seek loop)
    t_sig_ppr = np.array([t_sig[np.argmin(np.abs(t_sig - t))] for t in t_tach_ppr])
    t_sig_ind = np.array([np.where(t_sig == t)[0][0] for t in t_sig_ppr]).astype(int)

    # Interpolate fixed number of points between each revolution boundary
    t_cot   = []
    sig_cot = []

    for i in range(len(t_sig_ppr) - 1):
        t_start = t_sig[t_sig_ind[i]]
        t_end   = t_sig[t_sig_ind[i + 1]]
        t_rsmp  = np.linspace(t_start, t_end, orders)
        sig_int = np.interp(t_rsmp, t_sig, vibration)
        t_cot.append(t_rsmp)
        sig_cot.append(sig_int)

    t_cot   = np.array(t_cot).reshape(-1)
    sig_cot = np.array(sig_cot).reshape(-1)

    return t_cot, sig_cot
