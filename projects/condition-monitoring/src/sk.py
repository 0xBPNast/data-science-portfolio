import numpy as np
import scipy as sp


def sk(signal, Fs, Nw, No, win_type='hann'):
    """
    Spectral Kurtosis via STFT.

    Computes the kurtosis of each frequency bin across STFT frames to
    identify frequency bands with high impulsivity. Accepts arrays of
    window lengths and overlaps for multi-window comparison.

    Parameters
    ----------
    signal   : array — input signal (1D)
    Fs       : float — sampling frequency [Hz]
    Nw       : list  — window length(s) in samples
    No       : list  — overlap length(s) in samples (must match length of Nw)
    win_type : str   — window type (default 'hann')

    Returns
    -------
    results : list of tuples — [(freqs, speckurt), ...] for each window length

    Example
    -------
    results = sk(signal, Fs, Nw=[32, 64], No=[16, 32])
    freqs, speckurt = results[0]
    """
    results = []

    for n_points, n_over in zip(Nw, No):

        sfreq, _, stft = sp.signal.stft(
            signal, fs=Fs, window=win_type,
            nperseg=n_points, noverlap=n_over,
            return_onesided=True
        )

        speckurt = np.zeros(len(sfreq))
        for j in range(len(sfreq)):
            speckurt[j] = (
                np.mean(np.abs(stft[j, :]) ** 4) /
                np.mean(np.abs(stft[j, :]) ** 2) ** 2
            ) - 2

        results.append((sfreq, speckurt))

    return results
