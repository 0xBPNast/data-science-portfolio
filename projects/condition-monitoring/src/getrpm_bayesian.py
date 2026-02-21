import numpy as np
import scipy.signal as sig
from scipy.sparse.linalg import spsolve


def maketime(X, Fs):
    """
    Generate a time array for a signal of length X sampled at Fs.

    Parameters
    ----------
    X  : array — signal
    Fs : float — sampling frequency [Hz]

    Returns
    -------
    t : array — time array [s]
    """
    t0 = 0
    t1 = len(X) / Fs
    t  = np.arange(t0, t1 + 1 / Fs, 1 / Fs)
    return t


def PerformBayesianGeometryCompensation(t, N, M, e=[], beta=10.0e10, sigma=10.0):
    """
    Bayesian Geometry Compensation for shaft encoder imperfections.

    Estimates the true circumferential distances of encoder segments using
    Bayesian linear regression over M complete revolutions. Corrects for
    non-uniform encoder segment spacing caused by manufacturing imperfections,
    which would otherwise introduce spurious spikes into the RPM profile.

    Parameters
    ----------
    t     : array — zero crossing times (must have exactly M*N + 1 elements)
    N     : int   — number of encoder sections (PPR)
    M     : int   — number of complete revolutions
    e     : array — initial encoder geometry estimate (empty = assume equal)
    beta  : float — precision of likelihood function
    sigma : float — standard deviation of prior probability

    Returns
    -------
    epost : array — estimated circumferential distances for all N sections

    References
    ----------
    DH Diamond et al. (2015). Online shaft encoder geometry compensation for
    arbitrary shaft speed profiles using Bayesian regression. Centre for Asset
    Integrity Management, University of Pretoria.
    """
    if len(t) != M * N + 1:
        print('Input Error: t should contain exactly N*M + 1 values')
        raise SystemExit
    if len(e) != 0 and len(e) != N:
        print('Input Error: encoder input should be empty or length N')
        raise SystemExit

    # Build constraint matrices
    A = np.zeros((2 * M * N - 1, N + 2 * M * N))
    B = np.zeros((2 * M * N - 1, 1))
    T = np.ediff1d(t)   # Zero-crossing periods

    # Constraint: all segments sum to 2π
    A[0, :N] = np.ones(N)
    B[0, 0]  = 2 * np.pi

    # Insert continuity constraints (Equations 9 & 10)
    deduct = 0
    for m in range(M):
        if m == M - 1:
            deduct = 1
        for n in range(N - deduct):
            nm = m * N + n
            A[1 + nm, n]                      = 3.0
            A[1 + nm, N + nm * 2]             = -0.5 * T[nm] ** 2
            A[1 + nm, N + nm * 2 + 1]         = -2 * T[nm]
            A[1 + nm, N + (nm + 1) * 2 + 1]   = -T[nm]

    deduct = 0
    for m in range(M):
        if m == M - 1:
            deduct = 1
        for n in range(N - deduct):
            nm = m * N + n
            A[M * N + nm, n]              = 6.0
            A[M * N + nm, N + nm * 2]     = -2 * T[nm] ** 2
            A[M * N + nm, N + (nm+1)*2]   = -T[nm] ** 2
            A[M * N + nm, N + nm * 2 + 1] = -6 * T[nm]

    # Prior distribution
    m0     = np.zeros((N + 2 * M * N, 1))
    Sigma0 = np.identity(N + 2 * M * N) * sigma ** 2
    eprior = np.ones(N) * 2 * np.pi / N if len(e) == 0 else np.array(e) * 1.0
    m0[:N, 0] = eprior
    for m in range(M):
        for n in range(N):
            nm = m * N + n
            m0[N + nm * 2 + 1, 0] = m0[n, 0] / T[nm]

    # Bayesian posterior solve
    SigmaN = Sigma0 + beta * A.T.dot(A)
    BBayes = Sigma0.dot(m0) + beta * A.T.dot(B)
    mN     = np.array([spsolve(SigmaN, BBayes)]).T

    # Normalise to sum to 2π
    epost = mN[:N, 0] * 2 * np.pi / np.sum(mN[:N, 0])
    return epost


def getrpm(tacho, Fs, trig_level, slope, pprm, new_sample_freq):
    """
    Shaft RPM estimation with Bayesian Geometry Compensation.

    Estimates instantaneous shaft speed from a tachometer signal, correcting
    for encoder geometry imperfections via BCG. Outputs a smoothed, uniformly
    resampled RPM profile.

    Parameters
    ----------
    tacho           : array — tachometer signal (1D)
    Fs              : float — tachometer sampling frequency [Hz]
    trig_level      : float — pulse trigger level
    slope           : int   — positive (1) or negative (-1) pulse slope
    pprm            : int   — pulses per revolution
    new_sample_freq : float — output RPM resampling frequency [Hz]

    Returns
    -------
    trpm          : array — RPM time array [s]
    rpm           : array — instantaneous RPM
    spacing_store : array — BCG-corrected encoder spacing
    """
    if type(tacho) == list:
        tacho = np.array(tacho)

    y  = np.sign(tacho - trig_level)
    dy = np.diff(y)
    tt = maketime(dy, Fs)

    # Detect pulse edges
    pos = np.nonzero(dy > 0.8) if slope > 0 else np.nonzero(dy < -0.8)
    yt  = tt[pos]
    dt  = np.diff(yt)
    dt  = np.hstack([dt, np.array([dt[-1]])])

    # Get tachometer zero-crossing times for BCG
    t_eval     = np.arange(0, len(dy) / Fs + 1 / Fs, 1 / Fs)
    cross_ind  = np.where(tacho == trig_level)[0]
    t_cross    = t_eval[cross_ind]
    n_revs     = int(len(cross_ind) / pprm)
    t_sections = t_cross[:n_revs * pprm + 1]

    # Run BCG to get corrected encoder spacing
    spacing_rev = PerformBayesianGeometryCompensation(
        t_sections, pprm, n_revs, e=[], beta=10.0e10, sigma=10.0
    )

    # Map BCG spacing back across full signal
    spacing_store = np.zeros(len(dt))
    s_cnt = 0
    for i in range(len(dt) - 1):
        spacing_store[i] = spacing_rev[s_cnt]
        s_cnt = 0 if s_cnt == pprm - 1 else s_cnt + 1

    # Compute RPM from BCG-corrected spacing
    rpm = (60 / (2 * np.pi)) * (spacing_store / dt)

    # Smooth RPM
    b   = [0.25, 0.5, 0.25]
    rpm = sig.filtfilt(b, 1, rpm)

    # Resample RPM to uniform time grid
    N    = int(np.max(tt) * new_sample_freq + 1)
    trpm = np.linspace(0, np.max(tt), N)
    rpm  = np.interp(trpm, yt, rpm)

    # Remove NaN values
    valid = ~np.isnan(rpm)
    return trpm[valid], rpm[valid], spacing_store
