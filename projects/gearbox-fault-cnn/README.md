# Gearbox Fault Classification
### Signal Processing & Deep Learning Framework

This project presents an end-to-end framework for automated gearbox fault classification using a combination of physics-based synthetic data generation and a 1D Convolutional Neural Network (1D-CNN). The work was completed as part of an Honours thesis in Mechanical Engineering at the University of Pretoria (2024).

The central question explored: **does signal domain — time, frequency, angular, or order — meaningfully affect a CNN's ability to classify gearbox faults, and does COT pre-processing improve generalisation under varying operating conditions?**

---

## Project Overview

Gearboxes are critical components in heavy industrial equipment. In the South African mining industry specifically, the introduction of Battery Electric Vehicles and Collision Avoidance Systems has significantly increased driveline fatigue rates — making early fault detection essential and costly failures more common.

This project addresses the data availability bottleneck common to deep learning-based fault diagnostics by generating synthetic vibration data using a validated physics-based model, then training a 1D-CNN across multiple signal representations to assess classification performance and generalisation capability.

---

## Framework

```
gearbox-fault-cnn/
├── README.md
├── notebooks/
│   ├── 01_data_generation.ipynb        # 8-DOF LPDM simulation, Newmark-Beta solver
│   ├── 02_signal_processing.ipynb      # COT, domain transformations, preprocessing
│   └── 03_cnn_training_evaluation.ipynb # 1D-CNN training, results, generalisation
├── src/
│   ├── lpdm.py                         # 8-DOF dynamic model & Newmark-Beta solver
│   ├── tvms.py                         # Time-Varying Meshing Stiffness & fault induction
│   ├── motor.py                        # Squirrel cage motor torque-speed model
│   ├── preprocessing.py                # Segmentation, augmentation, standardisation
│   └── cnn.py                          # 1D-CNN architecture
└── data/
    └── README.md                       # Data generation instructions
```

---

## Part 1 — Synthetic Data Generation

A **Lumped-Parameter Dynamic Model (LPDM)** of a single-stage spur gearbox is implemented in Python, based on the 8-DOF model from Mohammed et al. (2015) and parameterised from Chaari et al. (2012). The model captures translational and rotational motion of the motor, pinion, gear, and load.

**System parameters:**

| Parameter | Value |
|---|---|
| Gear ratio | 2:1 (20 pinion teeth, 40 gear teeth) |
| Pinion radius | 28.19 mm |
| Gear radius | 56.38 mm |
| Pressure angle | 20° |
| Contact ratio | 1.6 |
| Synchronous speed | 1500 RPM |
| Motor type | Squirrel cage induction |

**Equations of motion** are solved using a **Newmark-Beta implicit integration scheme** (β=0.25, γ=0.5), producing displacement, velocity, and acceleration outputs for all 8 DOF. The pinion y-direction acceleration is selected as the vibration signal, emulating an accelerometer measurement.

**Operating conditions simulated:**

| Dataset | Load Type | Fault |
|---|---|---|
| 1_C_H | Constant (−20 Nm) | None |
| 2_C_CT | Constant | Cracked Tooth (10%, tooth 5) |
| 3_C_WT | Constant | Worn Teeth (10%, teeth 4-6) |
| 4_V_H | Varying (−10sin(2πt)+30 Nm) | None |
| 5_V_CT | Varying | Cracked Tooth |
| 6_V_WT | Varying | Worn Teeth |

---

## Part 2 — Fault Introduction via TVMS

Gear faults are introduced through a **Time-Varying Meshing Stiffness (TVMS)** model. The TVMS is implemented as a rectangular step function (Chaari et al., 2012) — computationally efficient while still capturing the key fault dynamics.

**Tooth crack** — stiffness reduction localised to a single tooth, proportional to crack severity:

$$k_{crack} = k_{healthy}(t)\left(1 - \frac{a(t)}{a_{critical}}\right)$$

**Worn teeth** — stiffness reduction distributed across multiple adjacent teeth. A 10% reduction is applied to represent early-stage faults — specifically chosen to assess the CNN's capability to detect faults before significant progression.

---

## Part 3 — Signal Processing & Domain Transformation

Raw time-domain data is processed through the following pipeline before CNN input:

1. **Sanitisation** — first second removed to eliminate motor run-up transients
2. **Downsampling** — from 200 kHz to target sampling frequencies (1024–6144 Hz in 1024 Hz steps) with anti-aliasing low-pass filter applied forward and reverse to prevent phase distortion
3. **Segmentation** — signal divided into fixed-length segments (input samples for CNN)
4. **Data augmentation** — additive Gaussian noise (0–20% of signal amplitude) applied to each segment to synthetically increase dataset size and improve noise robustness
5. **Domain transformation** — each segment transformed to produce four parallel data pipelines:
   - Time domain (raw)
   - Frequency domain (FFT)
   - Angular domain (COT resampled)
   - Order domain (FFT of COT signal)
6. **Standardisation** — zero mean, unit standard deviation applied per dataset

**Labels:** Healthy = 0, Cracked Tooth = 1, Worn Teeth = 2

**Train / Validation / Test split:** 80% / 10% / 10%

---

## Part 4 — 1D-CNN Architecture

The CNN architecture follows Jing et al. (2017), adapted for 3-class classification.

| Layer | Parameters |
|---|---|
| Input | 4096 points |
| Convolution | 10 filters, width 32, ReLU activation |
| Max Pooling | Sub-sampling rate 2 |
| Dense | 90 nodes, ReLU activation |
| Classification | 3 outputs, Softmax |

**Training configuration:**

| Parameter | Value |
|---|---|
| Optimiser | SGD |
| Learning rate | 0.03 |
| Momentum | 0.5 |
| Weight decay | 0.02 |
| Batch size | 30 |
| Max epochs | 100 |
| Loss function | Cross-entropy |

Each model is trained over **10 independent runs** with weight reset between runs to assess stability.

---

## Results

### Classification Accuracy by Domain (4096 segment length, mean ± std over 10 runs)

| Domain | Accuracy |
|---|---|
| **Frequency** | **96% ± 4%** |
| Order (COT-Frequency) | 93% ± 3% |
| Angular (COT) | 90% ± 5% |
| Time | 55% ± 11% |

The **frequency domain consistently outperformed** all other representations, with strong stability across runs. The order-tracked signal (COT-frequency) was closely competitive at larger segment lengths, converging to within 1% of the frequency domain at 6144 points.

The time domain performed poorly at all but the largest segment lengths — consistent with the established view that raw vibration signals require transformation before fault features become identifiable.

### Key Findings

**Healthy vs tooth crack misclassification** was the primary challenge across all domains. A 10% crack severity combined with additive noise caused the CNN to conflate healthy and cracked tooth signals — suggesting the model learned noise characteristics rather than true fault features. Removing additive noise eliminated this misclassification entirely in the frequency and angular domains, confirming this hypothesis.

**Generalisation was limited.** When tested on data from different operating conditions, all models achieved only 33–37% accuracy. On a real-world dataset from the 2023 PHM Data Challenge, accuracy dropped to 0%. This indicates the models learned amplitude and load-dependent features rather than fault-specific signatures — a known limitation of training exclusively on synthetic data from a single operating regime.

### Honest Assessment

The results are directionally promising but reveal a fundamental challenge: a model that achieves 96% accuracy on held-out data from the same distribution can fail completely on data from a different distribution. This generalisation gap is the most important finding of the study and defines the primary direction for future work.

---

## Future Work

- **Data diversity** — training across a wider range of operating conditions and fault severities to prevent load-amplitude overfitting
- **Transfer learning** — fine-tuning on small amounts of real-world data to bridge the synthetic-to-real domain gap
- **Digital Twin integration** — using the validated LPDM as a real-time physics counterpart for anomaly detection
- **Higher fault severities** — investigating whether 10% crack severity is below a practical detection threshold for 1D-CNNs without signal processing pre-amplification

---

## Dependencies

```
numpy
scipy
matplotlib
torch          # or tensorflow
pandas
scikit-learn
```

---

## Background

This project forms part of a broader portfolio of work on vibration-based condition monitoring and machine learning for rotating machinery diagnostics. For the classical signal processing foundation this work builds on — including TSA, Spectral Kurtosis, Envelope Analysis, Bayesian Geometry Compensation and Computed Order Tracking — see the [condition monitoring project](../condition-monitoring/).

**Thesis:** *An Investigative Study into Gearbox Fault Classification Using a Signal Processing and Deep Learning Framework*, Bradley Nast, University of Pretoria, 2024.  
**Supervisors:** Prof. PS Heyns, Luke van Eyk