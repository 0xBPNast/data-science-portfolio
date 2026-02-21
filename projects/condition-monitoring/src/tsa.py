import numpy as np


def tsa(tacho, Fs_tacho, signal, Fs_signal, tacho_trigger, tacho_ppr):
    """
    Time Synchronous Averaging (TSA).

    Isolates the deterministic (gear) component of a vibration signal by
    averaging over complete shaft revolutions, synchronised to a tachometer.
    Handles both equal and unequal pulse spacing via interpolation.

    Parameters
    ----------
    tacho         : array — tachometer signal (1D)
    Fs_tacho      : float — tachometer sampling frequency [Hz]
    signal        : array — vibration signal (1D)
    Fs_signal     : float — vibration sampling frequency [Hz]
    tacho_trigger : float — tachometer pulse trigger value
    tacho_ppr     : int   — pulses per revolution

    Returns
    -------
    method      : str   — 'Reinterpolated' or 'Standard'
    x_tsa_nr    : array — TSA signal extended over full signal length
    Ns          : int   — samples per revolution
    Ns_tot      : int   — total samples across all revolutions
    Nr          : int   — number of complete revolutions averaged
    t_tacho     : array — tacho time array
    ind_ppr_tac : array — PPRth tacho pulse indices
    t_signal    : array — vibration signal time array
    ind_ppr_sig : array — PPRth vibration signal indices
    """

    # ── Tachometer signal parameters ─────────────────────────────────────────
    dt_tacho = 1 / Fs_tacho
    N_tacho  = np.size(tacho)
    tf_tacho = dt_tacho * N_tacho
    t_tacho  = np.arange(0, tf_tacho, dt_tacho)

    # Find sample indices where tacho pulse occurs
    ind_tac = np.where(tacho == tacho_trigger)[0]

    # Extract every PPRth pulse index (one per revolution)
    Nr          = int(np.floor(len(ind_tac) / tacho_ppr))
    ind_ppr_tac = np.zeros(Nr, dtype=int)
    for i in range(1, Nr + 1):
        ind_ppr_tac[i - 1] = ind_tac[tacho_ppr * i - 1]

    # Get time points at each revolution boundary
    t_tacho_pprs = t_tacho[ind_ppr_tac]

    # ── Vibration signal parameters ───────────────────────────────────────────
    dt_signal = 1 / Fs_signal
    N_signal  = np.size(signal)
    tf_signal = dt_signal * N_signal
    t_signal  = np.arange(0, tf_signal, dt_signal)

    # Correlate tacho revolution times with vibration signal indices
    ind_ppr_sig = np.zeros(len(ind_ppr_tac), dtype=int)
    for i in range(len(t_tacho_pprs)):
        ind_ppr_sig[i] = np.argmin(np.abs(t_signal - t_tacho[ind_ppr_tac[i]]))

    # Check whether sample count between revolutions is uniform
    diff_check = np.diff(ind_ppr_sig)

    # ── Next power of 2 helper ────────────────────────────────────────────────
    def nextpower(value):
        power = 2 ** np.arange(0, 16, 1)
        for p in power:
            if value < p:
                return p

    # ── Case 1: Unequal spacing — resample each revolution to fixed length ────
    if np.mean(diff_check) != 0:

        Ns     = nextpower(diff_check[0])
        Ns_tot = Ns * Nr
        t_ns   = []
        s_ns   = []
        x_tsa  = np.zeros(Ns)

        for i in range(Nr):
            t_start = 0 if i == 0 else t_signal[ind_ppr_sig[i - 1]]
            t_end   = t_signal[ind_ppr_sig[i]]
            t_int   = np.linspace(t_start, t_end, Ns)
            sig_int = np.interp(t_int, t_signal, signal)
            t_ns.append(t_int)
            s_ns.append(sig_int)
            x_tsa += sig_int

        t_ns  = np.array(t_ns).reshape(-1)
        s_ns  = np.array(s_ns).reshape(-1)
        x_tsa = x_tsa / Nr

        # Extend TSA over full signal length
        x_tsa_nr = np.zeros(len(t_ns))
        for i in range(Nr):
            x_tsa_nr[i * Ns:(i + 1) * Ns] = x_tsa

        x_tsa_nr = np.interp(t_signal[:ind_ppr_sig[-1]], t_ns, x_tsa_nr)

        return ('Reinterpolated', x_tsa_nr, Ns, Ns_tot, Nr,
                t_tacho, ind_ppr_tac, t_signal, ind_ppr_sig)

    # ── Case 2: Equal spacing — direct averaging ──────────────────────────────
    else:

        Ns     = diff_check[0]
        Ns_tot = Ns * Nr
        x_tsa  = np.zeros(Ns)

        for i in range(Nr):
            start  = ind_ppr_sig[i]
            x_tsa += signal[start:start + Ns]
        x_tsa = x_tsa / Nr

        # Extend TSA over full signal length
        x_tsa_nr = np.zeros(Nr * Ns)
        for i in range(Nr):
            x_tsa_nr[i * Ns:(i + 1) * Ns] = x_tsa

        return ('Standard', x_tsa_nr, Ns, Ns_tot, Nr,
                t_tacho, ind_ppr_tac, t_signal, ind_ppr_sig)
