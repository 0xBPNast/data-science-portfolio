# Vibration-Based Condition Monitoring
### Signal Processing & Fault Diagnostics in Python

This project contains a suite of custom Python implementations for vibration-based condition monitoring (CM) of rotating machinery — specifically gearboxes and bearings. The work covers the full diagnostic pipeline from raw signal acquisition through to fault identification, including both classical signal processing and learning-based approaches.

The code was developed as part of the MEV781 module (Vibration Based Condition Monitoring) at the University of Pretoria, Faculty of Engineering.

---

## Contents

```
condition-monitoring/
├── notebooks/
│   ├── 01_static_load_analysis.ipynb       # TSA, Spectral Kurtosis, Envelope Analysis
│   ├── 02_varying_load_analysis.ipynb      # BCG, Order Tracking, SES
│   └── 03_learning_based_diagnostics.ipynb # Hankel Matrix, HI & LHI via PCA
├── src/
│   ├── tsa.py                              # Time Synchronous Averaging
│   ├── spectral_kurtosis.py                # Custom SK implementation
│   ├── envelope_analysis.py                # Hilbert transform & SES
│   ├── order_tracking.py                   # Computed Order Tracking (COT)
│   ├── bcg.py                              # Bayesian Geometry Compensation
│   └── health_indicators.py               # Hankel matrix, HI & LHI
├── data/                                   # Sample signals (see Data section)
└── README.md
```

---

## Diagnostic Pipeline

### Part 1 — Static Load Analysis

Covers gearbox fault diagnostics under constant shaft speed (900 RPM).

**Signal decomposition** — A phenomenological gearbox model is used to understand the individual contributions of gear mesh vibration, bearing fault vibration, and broadband noise within a composite gearbox signal.

**Time Synchronous Averaging (TSA)** — A custom TSA function isolates the deterministic (gear) signal component by averaging over complete shaft revolutions. The residual signal `x_residual = x_gearbox - x_TSA` isolates the stochastic (bearing + noise) component for downstream analysis.

**Spectral Kurtosis (SK)** — A custom STFT-based SK function identifies the frequency band of highest impulsivity within the residual signal, guiding passband filter selection. Window length trade-offs are explored across a range of 16 to 4096 samples.

**Envelope Analysis** — A Butterworth bandpass filter (informed by SK) is applied to the residual signal. The Hilbert transform yields the signal envelope, and frequency analysis of the Squared Envelope Spectrum (SES) cleanly identifies the Ball Pass Frequency Outer Race (BPFO) and its harmonics.

**Key result:** BPFO identified at 53.5 Hz with harmonics clearly visible in the SES. Gear tooth fault isolated via TSA amplitude peaks at specific tooth indices.

---

### Part 2 — Varying Load Analysis

Covers fault diagnostics under sinusoidally varying shaft speed (~500–2000 RPM), where classical FFT analysis is insufficient due to frequency smearing.

**Bayesian Geometry Compensation (BCG)** — Shaft encoder imperfections (non-uniform segment spacing) are corrected using Bayesian linear regression, significantly smoothing the RPM profile and improving downstream analysis accuracy.

**Computed Order Tracking (COT)** — A custom order tracking function resamples the vibration signal to the angular domain (equal angle increments), converting time-varying frequency content into fixed order content. This eliminates smearing of the Gear Meshing Frequency (GMF) harmonics.

**Order Spectrum & SES Analysis** — Following COT, the SES reveals the GMF fundamental at order 30 and harmonics up to order 480. Bearing outer race fault orders (BPOO = 3.57) are also identified in the varying-speed signal.

**Key result:** GMF sidebands and bearing fault orders cleanly identified in a varying-speed gearbox signal that could not be diagnosed with standard FFT analysis.

---

### Part 3 — Learning-Based Diagnostics

A comparison of classical signal processing against a learning-based approach using linear algebra (no neural network required).

**Hankel Matrix Construction** — Raw vibration signals are structured into Hankel matrices to exploit the time-delayed embedding of the signal.

**Health Indicator (HI) & Latent Health Indicator (LHI)** — Principal Component Analysis (PCA) is applied to the Hankel matrix representation to derive a Health Indicator. The LHI extends this with additional dimensionality reduction, amplifying the signal-to-noise ratio and preserving both bearing fault frequencies and GMF content.

**Comparison: Signal Processing vs. Learning-Based**

| Step | Signal Processing | Learning-Based |
|---|---|---|
| Signal Separation | TSA — requires tachometer, sensitive to speed variation | Not required for bearing analysis (use HI directly) |
| Fault Enhancement | Spectral Kurtosis — window selection required | Inherent in PCA covariance decomposition |
| Envelope | Bandpass filter + Hilbert transform | HI/LHI acts as the signal envelope |

**Key result:** The learning-based approach achieves comparable fault identification with fewer processing steps and lower computational overhead — well suited to near-real-time applications.

---

## Key Techniques Implemented

All functions are custom Python implementations, not wrappers around existing CM libraries.

- Time Synchronous Averaging (TSA)
- Spectral Kurtosis via Short-Time Fourier Transform (STFT)
- Butterworth bandpass filtering
- Hilbert transform & Squared Envelope Spectrum (SES)
- Bayesian Geometry Compensation (BCG)
- Computed Order Tracking (COT)
- Hankel matrix construction
- PCA-based Health Indicators (HI & LHI)

---

## System Parameters

The analysis is performed on a simulated gearbox system based on the SKF 6206 bearing:

| Parameter | Value |
|---|---|
| Shaft speed (static) | 900 RPM |
| Gear teeth | 30 |
| Gear meshing frequency | 450 Hz |
| BPFO | 53.52 Hz |
| BPFI | 81.48 Hz |
| FTF | 5.95 Hz |
| BSF | 34.67 Hz |

---

## Data

Sample `.mat` / `.csv` signal files are provided in the `data/` directory for reproducing all results. These include:
- Tachometer signals (constant and varying speed)
- Composite gearbox vibration signals
- Healthy and unhealthy bearing signals for HI/LHI analysis

---

## Dependencies

```
numpy
scipy
matplotlib
pandas
```

Install with:
```bash
pip install numpy scipy matplotlib pandas
```

---

## Background

This work was completed as part of a BEng (Hons) in Mechanical Engineering at the University of Pretoria. It forms part of a broader portfolio of work on machine learning for rotating machinery fault diagnostics — see the [gearbox fault CNN project](../gearbox-fault-cnn/) for the deep learning extension of this work.