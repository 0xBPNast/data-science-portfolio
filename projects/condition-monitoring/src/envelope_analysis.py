import numpy as np
from scipy.signal import butter, filtfilt, hilbert
from scipy.fft import fft, fftfreq


def envelope_analysis(x, fs, f_low, f_high, filter_order=4):
    """
    Bandpass filter + Hilbert envelope extraction.

    Applies a Butterworth bandpass filter to isolate the frequency band of
    interest (determined via Spectral Kurtosis), then uses the Hilbert
    transform to extract the signal envelope for bearing fault identification.

    Parameters
    ----------
    x            : array — input signal (residual)
    fs           : float — sampling frequency [Hz]
    f_low        : float — passband lower bound [Hz]
    f_high       : float — passband upper bound [Hz]
    filter_order : int   — Butterworth filter order (default 4)

    Returns
    -------
    x_filtered : array — bandpass filtered signal
    envelope   : array — signal envelope (DC removed)
    """
    nyq  = fs / 2
    b, a = butter(filter_order, [f_low / nyq, f_high / nyq], btype='band')
    x_filtered = filtfilt(b, a, x)

    # Hilbert transform → analytic signal → envelope
    analytic  = hilbert(x_filtered)
    envelope  = np.abs(analytic)
    envelope -= np.mean(envelope)   # Remove DC component

    return x_filtered, envelope


def compute_ses(envelope, fs):
    """
    Squared Envelope Spectrum (SES).

    Computes the frequency spectrum of the squared signal envelope.
    Bearing fault repetition frequencies (BPFO, BPFI, etc.) appear
    as clear peaks in the SES.

    Parameters
    ----------
    envelope : array — signal envelope (DC removed)
    fs       : float — sampling frequency [Hz]

    Returns
    -------
    freqs : array — frequency axis [Hz]
    ses   : array — SES magnitude (single-sided)
    """
    N       = len(envelope)
    ses_fft = np.abs(fft(envelope ** 2)) / N
    freqs   = fftfreq(N, 1 / fs)
    half    = N // 2
    return freqs[:half], ses_fft[:half]
