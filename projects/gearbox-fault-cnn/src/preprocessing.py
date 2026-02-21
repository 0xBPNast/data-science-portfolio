"""
preprocessing.py — Signal Preprocessing Pipeline

Implements the full preprocessing chain for converting raw LPDM simulation
output into CNN-ready input across four signal domains:

    Time domain     — downsampled vibration signal
    Frequency domain — FFT magnitude spectrum
    Angular domain  — Computed Order Tracking (COT) resampled signal
    Order domain    — FFT of COT signal

Pipeline steps:
    1. Anti-aliasing filter + downsample
    2. Sanitisation (remove motor run-up transient)
    3. COT resampling using synthetic or real tachometer
    4. Segmentation into fixed-length windows
    5. Gaussian noise augmentation
    6. Domain transformation (FFT)
    7. Label merging (constant + varying load per fault class)
"""

import numpy as np
from scipy.signal import butter, filtfilt, resample
from scipy.fft import fft


# ── COT ────────────────────────────────────────────────────────────────────────

def cot(tach, Fs_tach, ppr, trigger, vibration, Fs_vibration, orders):
    """
    Computed Order Tracking (COT).

    Resamples a vibration signal to the angular domain by interpolating
    a fixed number of evenly-spaced points per shaft revolution. Eliminates
    frequency smearing caused by varying shaft speed.

    Parameters
    ----------
    tach         : array — tachometer signal
    Fs_tach      : float — tachometer sampling frequency [Hz]
    ppr          : int   — pulses per revolution
    trigger      : float — pulse detection threshold
    vibration    : array — vibration signal to resample
    Fs_vibration : float — vibration sampling frequency [Hz]
    orders       : int   — resampled points per revolution

    Returns
    -------
    t_cot   : array — resampled time array
    sig_cot : array — order-tracked vibration signal
    """
    dt_tach       = 1 / Fs_tach
    N_tach        = len(tach)
    t_tach        = np.linspace(0, N_tach * dt_tach, N_tach)
    pulse_indices = np.where(tach >= trigger)[0]
    ppr_indices   = pulse_indices[::ppr]

    t_cot, sig_cot = [], []
    for i in range(len(ppr_indices) - 1):
        t_start   = t_tach[ppr_indices[i]]
        t_end     = t_tach[ppr_indices[i + 1]]
        t_order   = np.linspace(t_start, t_end, orders)
        sig_order = np.interp(t_order,
                              np.arange(len(vibration)) / Fs_vibration,
                              vibration)
        t_cot.extend(t_order)
        sig_cot.extend(sig_order)

    return np.array(t_cot), np.array(sig_cot)


def generate_tachometer(angular_velocity, Fs, pulses_per_revolution=60):
    """
    Generate a synthetic tachometer signal from angular velocity.

    Produces a pulse train where each pulse marks the passage of a
    tooth/encoder mark, derived from cumulative angular displacement.

    Parameters
    ----------
    angular_velocity     : array — shaft angular velocity [rad/s]
    Fs                   : float — sampling frequency [Hz]
    pulses_per_revolution: int   — encoder resolution (default 60)

    Returns
    -------
    tachometer_signal : array — binary pulse train (same length as input)
    """
    dt               = 1 / Fs
    cumulative_angle = np.cumsum(angular_velocity * dt)
    pulse_interval   = 2 * np.pi / pulses_per_revolution
    tachometer       = np.zeros_like(angular_velocity)
    pulse_count      = 0

    for i in range(1, len(cumulative_angle)):
        if cumulative_angle[i] >= (pulse_count + 1) * pulse_interval:
            tachometer[i] = 1
            pulse_count  += 1

    return tachometer


# ── Signal Processing Utilities ────────────────────────────────────────────────

def downsample_signal(signal, Fs_original, Fs_target, filter_order=10):
    """
    Anti-aliasing low-pass filter followed by resampling.

    Uses a forward-reverse Butterworth filter (filtfilt) to prevent
    phase distortion. Cutoff set to 95% of target Nyquist frequency.

    Parameters
    ----------
    signal      : array — input signal
    Fs_original : float — original sampling rate [Hz]
    Fs_target   : float — target sampling rate [Hz]
    filter_order: int   — Butterworth filter order (default 10)

    Returns
    -------
    resampled : array — downsampled signal
    """
    nyquist     = Fs_target / 2
    cutoff      = nyquist * 0.95
    b, a        = butter(N=filter_order, Wn=cutoff / (Fs_original / 2), btype='low')
    filtered    = filtfilt(b, a, signal)
    n_target    = int(len(filtered) / (Fs_original / Fs_target))
    return resample(filtered, n_target)


def segment_signal(signal, segment_length, overlap=0.0):
    """
    Divide a signal into fixed-length segments with optional overlap.

    Parameters
    ----------
    signal         : array — 1D signal
    segment_length : int   — length of each segment [samples]
    overlap        : float — fractional overlap between segments (0.0–1.0)

    Returns
    -------
    segments : array (n_segments x segment_length)
    """
    step_size = int(segment_length * (1 - overlap))
    n_seg     = (len(signal) - segment_length) // step_size + 1
    return np.array([
        signal[i * step_size: i * step_size + segment_length]
        for i in range(n_seg)
    ])


def augment_with_noise(segments, snr_db_lower, snr_db_upper, target_num):
    """
    Additive Gaussian noise augmentation.

    Synthetically expands the dataset by adding noise at random SNR
    levels within the specified range. SNR is computed relative to
    the RMS power of each segment.

    Parameters
    ----------
    segments      : array — input segments (n x segment_length)
    snr_db_lower  : float — minimum SNR [dB]
    snr_db_upper  : float — maximum SNR [dB]
    target_num    : int   — desired total number of augmented segments

    Returns
    -------
    augmented : array (target_num x segment_length)
    """
    augmented   = []
    n_aug_per   = max(1, target_num // len(segments))

    for segment in segments:
        for _ in range(n_aug_per):
            snr_db      = np.random.uniform(snr_db_lower, snr_db_upper)
            snr_linear  = 10 ** (snr_db / 10)
            sig_power   = np.mean(segment ** 2)
            noise_power = sig_power / snr_linear
            noise       = np.sqrt(noise_power) * np.random.randn(*segment.shape)
            augmented.append(segment + noise)

    return np.array(augmented[:target_num])


def to_frequency_domain(segments):
    """
    Transform segments to frequency domain via FFT magnitude.

    Parameters
    ----------
    segments : array (n x segment_length)

    Returns
    -------
    freq_segments : array (n x segment_length)
    """
    return np.array([np.abs(fft(seg)) for seg in segments])


# ── Full Preprocessing Pipeline ────────────────────────────────────────────────

def preprocess_vibration_data(vibration_data, rotation_data,
                               time_start, time_end,
                               original_sampling_rate, target_sampling_rate,
                               segment_length, target_num_segments,
                               snr_db_lower, snr_db_upper,
                               overlap=0.0):
    """
    Full preprocessing pipeline for 1D-CNN input.

    Accepts pre-loaded vibration and rotation arrays, downsamples,
    applies COT, segments, augments with Gaussian noise, transforms
    to four signal domains, and merges constant and varying load cases
    by fault class.

    Expected input order for both lists:
        [0] 1_C_H  — Constant load, Healthy
        [1] 2_C_CT — Constant load, Cracked Tooth
        [2] 3_C_WT — Constant load, Worn Teeth
        [3] 4_V_H  — Varying load, Healthy
        [4] 5_V_CT — Varying load, Cracked Tooth
        [5] 6_V_WT — Varying load, Worn Teeth

    Parameters
    ----------
    vibration_data         : list  — 6 arrays of pinion y-acceleration
    rotation_data          : list  — 6 arrays of pinion angular velocity
    time_start, time_end   : float — sanitised signal window [s]
    original_sampling_rate : float — simulation sampling rate [Hz]
    target_sampling_rate   : float — downsampled target rate [Hz]
    segment_length         : int   — CNN input length [samples]
    target_num_segments    : int   — augmented segments per dataset
    snr_db_lower/upper     : float — SNR range for noise augmentation [dB]
    overlap                : float — segment overlap fraction (default 0.0)

    Returns
    -------
    time_segments : array — time domain input
    freq_segments : array — frequency domain input
    all_labels    : array — class labels (0=Healthy, 1=CT, 2=WT)
    cot_segments  : array — angular domain (COT) input
    cotf_segments : array — order domain (COT-FFT) input
    cot_labels    : array — class labels (same as all_labels)
    """
    start_idx = int(time_start * target_sampling_rate)
    end_idx   = int(time_end   * target_sampling_rate)

    segmented_vib, segmented_cot = [], []

    for vib_data, rot_data in zip(vibration_data, rotation_data):
        # Downsample
        vib_ds = downsample_signal(vib_data, original_sampling_rate, target_sampling_rate)
        rot_ds = downsample_signal(rot_data, original_sampling_rate, target_sampling_rate)

        # Sanitise — remove motor run-up transient
        vib_cut = vib_ds[start_idx:end_idx]
        rot_cut = rot_ds[start_idx:end_idx]

        # Generate synthetic tachometer and apply COT
        tacho   = generate_tachometer(rot_cut, target_sampling_rate)
        _, sig_cot = cot(tach=tacho, Fs_tach=target_sampling_rate,
                         ppr=60, trigger=1,
                         vibration=vib_cut, Fs_vibration=target_sampling_rate,
                         orders=4096)

        # Pad COT output to match vibration length
        sig_cot = np.pad(sig_cot,
                         (0, max(0, len(vib_cut) - len(sig_cot))),
                         'constant')[:len(vib_cut)]

        segmented_vib.append(segment_signal(vib_cut, segment_length, overlap))
        segmented_cot.append(segment_signal(sig_cot, segment_length, overlap))

    # Augment and transform all six datasets
    all_vib, all_freq, all_cot, all_cotf, all_labels = [], [], [], [], []

    for idx, (vib_seg, cot_seg) in enumerate(zip(segmented_vib, segmented_cot)):
        aug_vib  = augment_with_noise(vib_seg, snr_db_lower, snr_db_upper, target_num_segments)
        aug_cot  = augment_with_noise(cot_seg, snr_db_lower, snr_db_upper, target_num_segments)

        all_vib.append(aug_vib)
        all_freq.append(to_frequency_domain(aug_vib))
        all_cot.append(aug_cot)
        all_cotf.append(to_frequency_domain(aug_cot))
        all_labels.extend([idx] * len(aug_vib))

    # Merge constant + varying load cases by fault class
    # (0,3) → Healthy, (1,4) → Cracked Tooth, (2,5) → Worn Teeth
    merged_vib, merged_freq, merged_cot, merged_cotf, merged_labels = [], [], [], [], []

    for health_state, (c_idx, v_idx) in enumerate([(0, 3), (1, 4), (2, 5)]):
        merged_vib.append( np.concatenate([all_vib[c_idx],  all_vib[v_idx]]))
        merged_freq.append(np.concatenate([all_freq[c_idx], all_freq[v_idx]]))
        merged_cot.append( np.concatenate([all_cot[c_idx],  all_cot[v_idx]]))
        merged_cotf.append(np.concatenate([all_cotf[c_idx], all_cotf[v_idx]]))
        merged_labels.extend(
            [health_state] * (len(all_vib[c_idx]) + len(all_vib[v_idx])))

    return (
        np.concatenate(merged_vib),
        np.concatenate(merged_freq),
        np.array(merged_labels),
        np.concatenate(merged_cot),
        np.concatenate(merged_cotf),
        np.array(merged_labels)
    )


def process_and_segment(vibration, angular_velocity_or_tachometer,
                         Fs_original, Fs_target,
                         time_start, time_end, segment_length,
                         use_real_tachometer=False):
    """
    Downsample, sanitise, COT-transform and segment a single signal
    into all four domain representations.

    Used for inference on unseen or real-world data (Notebook 3).

    Parameters
    ----------
    vibration                        : array — vibration signal
    angular_velocity_or_tachometer   : array — angular velocity [rad/s]
                                                or real tachometer signal
    Fs_original                      : float — original sampling rate [Hz]
    Fs_target                        : float — target sampling rate [Hz]
    time_start                       : float — start of analysis window [s]
    time_end                         : float — end of analysis window [s]
    segment_length                   : int   — CNN input length [samples]
    use_real_tachometer              : bool  — True if passing real tachometer
                                               (skips synthetic generation)

    Returns
    -------
    dict with keys 'time', 'freq', 'cot', 'cotf' — each an array of segments
    """
    vib_ds  = downsample_signal(vibration,                        Fs_original, Fs_target)
    rot_ds  = downsample_signal(angular_velocity_or_tachometer,   Fs_original, Fs_target)

    s_idx   = int(time_start * Fs_target)
    e_idx   = int(time_end   * Fs_target)
    vib_cut = vib_ds[s_idx:e_idx]
    rot_cut = rot_ds[s_idx:e_idx]

    # Generate or use tachometer for COT
    if use_real_tachometer:
        tacho = rot_cut
    else:
        tacho = generate_tachometer(rot_cut, Fs_target)

    _, sig_cot = cot(tach=tacho, Fs_tach=Fs_target,
                     ppr=60, trigger=1,
                     vibration=vib_cut, Fs_vibration=Fs_target,
                     orders=4096)

    sig_cot = np.pad(sig_cot,
                     (0, max(0, len(vib_cut) - len(sig_cot))),
                     'constant')[:len(vib_cut)]

    time_segs = segment_signal(vib_cut, segment_length)
    cot_segs  = segment_signal(sig_cot, segment_length)

    return {
        'time': time_segs,
        'freq': to_frequency_domain(time_segs),
        'cot' : cot_segs,
        'cotf': to_frequency_domain(cot_segs)
    }
