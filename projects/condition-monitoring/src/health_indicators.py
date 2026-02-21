import numpy as np


def Hankel_matrix(signal, Lw=512, Lsft=1):
    """
    Construct a Hankel matrix from a vibration signal.

    Slides a window of length Lw across the signal with step Lsft,
    storing each windowed segment as a row. The resulting matrix
    captures the time-delayed embedding of the signal and is used
    as input to PCA for health indicator derivation.

    Parameters
    ----------
    signal : array — input vibration signal (1D)
    Lw     : int   — window length in samples (default 512)
    Lsft   : int   — window shift/step in samples (default 1)

    Returns
    -------
    Hmat : array — Hankel matrix of shape (n_windows, Lw)
    """
    N    = len(signal)
    Lh   = int((N - Lw) / Lsft) + 1
    Hmat = np.zeros((Lh, Lw))

    for i in range(Lh):
        start      = int(i * Lsft)
        end        = int(Lw + i * Lsft)
        Hmat[i, :] = signal[start:end]

    return Hmat


def reconstruction_error(X, Xrecon):
    """
    Health Indicator (HI) via PCA reconstruction error.

    Computes the sum of squared differences between the original
    Hankel matrix rows and their PCA reconstructions. High error
    indicates signal content not captured by the healthy subspace,
    which is associated with fault-related vibration.

    Parameters
    ----------
    X      : array — original Hankel matrix (n_windows, Lw)
    Xrecon : array — PCA-reconstructed Hankel matrix (n_windows, Lw)

    Returns
    -------
    HI_error : array — reconstruction error per window (n_windows,)
    """
    ms       = np.subtract(X, Xrecon)
    HI_error = (ms ** 2).sum(axis=1)
    return HI_error


def latent_norm(Z):
    """
    Latent Health Indicator (LHI) via PCA latent space norm.

    Computes the L2 norm squared of each row in the PCA latent
    space. PCA amplifies the signal-to-noise ratio through
    covariance decomposition, making fault frequencies more
    visible in the LHI than in the raw HI.

    Parameters
    ----------
    Z : array — PCA latent space representation (n_windows, n_components)

    Returns
    -------
    LHI : array — latent norm per window (n_windows,)
    """
    LHI = (Z ** 2).sum(axis=1)
    return LHI
