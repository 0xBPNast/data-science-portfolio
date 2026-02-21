# Data Science Portfolio
### Bradley Nast

Mechanical engineer transitioning into data science, with a focus on applying machine learning to industrial condition monitoring and predictive maintenance. This portfolio documents that transition through two end-to-end projects built from original research and real engineering problems.

---

## About

**Background:** BEng Mechanical Engineering, University of Pretoria. Honours research in vibration-based gearbox fault classification (2024). Three years of practical experience in mining and heavy industry.

**Why data science:** I’m a problem solver at my core, and I enjoy owning the full loop: framing the problem, exploring the data, building models, and translating results into actions. Data science gives me leverage—my work can improve decisions, reduce risk, and create value across teams.

**Stack:** Python · NumPy · SciPy · TensorFlow/Keras · scikit-learn · pandas · Matplotlib · h5py

---

## Projects

### 1. Bearing & Gear Condition Monitoring
`projects/condition-monitoring/`

Classical signal processing pipeline for fault detection in a single-stage gearbox system (SKF 6206 bearing, 30-tooth gear). Demonstrates the signal processing fundamentals that underpin the deep learning project.

**Notebooks:**
| Notebook | Description |
|---|---|
| `01_static_load_analysis.ipynb` | TSA, kurtosis, crest factor, envelope analysis under constant load |
| `02_varying_load_analysis.ipynb` | Computed Order Tracking (COT) for speed-varying conditions |
| `03_learning_based_diagnostics.ipynb` | Feature extraction + classical ML classifiers (SVM, RF, kNN) |

**Source modules** (`src/`):
- `tsa.py` — Time Synchronous Averaging
- `sk.py` — Spectral Kurtosis
- `envelope_analysis.py` — Hilbert transform envelope detection
- `getrpm_bayesian.py` — Bayesian RPM estimation from tachometer
- `cot.py` — Computed Order Tracking resampling
- `health_indicators.py` — RMS, kurtosis, crest factor, peak-to-peak

---

### 2. Gearbox Fault Classification — CNN on Synthetic Data
`projects/gearbox-fault-cnn/`

End-to-end deep learning pipeline for automated gearbox fault classification. A physics-based dynamic model generates synthetic vibration data which trains a 1D-CNN across four signal domains. Generalisation to real-world data is tested against the PHM 2023 Data Challenge dataset.

Built from my Honours thesis: *An Investigative Study into Gearbox Fault Classification Using a Signal Processing and Deep Learning Framework* (University of Pretoria, 2024).

**Notebooks:**
| Notebook | Description |
|---|---|
| `01_data_generation.ipynb` | 8-DOF LPDM simulation, TVMS fault introduction, Newmark-Beta solver |
| `02_cnn_training.ipynb` | Preprocessing pipeline, 1D-CNN training across 4 domains × 6 segment lengths |
| `03_generalisation.ipynb` | Out-of-distribution testing — unseen conditions and PHM real-world data |

**Source modules** (`src/`):
- `lpdm.py` — 8-DOF Lumped Parameter Dynamic Model and Newmark-Beta solver
- `tvms.py` — Time-Varying Meshing Stiffness and squirrel cage motor dynamics
- `preprocessing.py` — Full signal preprocessing pipeline (downsample, COT, segment, augment)
- `cnn.py` — 1D-CNN architecture, training loop, normalisation utilities
- `simulation.py` — High-level simulation runner and HDF5 I/O

**Key results:**

| Domain | Best Accuracy (4096 samples) |
|---|---|
| Frequency | 96% ± 4% |
| COT-Frequency | 93% ± 3% |
| COT (Angular) | 90% ± 5% |
| Time | 55% ± 11% |

**Generalisation:**

| Test | Accuracy |
|---|---|
| Unseen synthetic conditions (different load, higher severity) | 33–37% |
| PHM 2023 real-world gearbox | 0% |

The 96% → 0% drop is the most important result. Models learned load-amplitude characteristics rather than fault-specific signatures, and failed completely when tested on real gearbox data. This defines the roadmap for future work.

**Data:** The `data/` folder is gitignored due to file size. Run `01_data_generation.ipynb` to regenerate all six synthetic datasets. The PHM 2023 dataset (`1_V1500_100N.txt`) is available from the [PHM Society](https://phmsociety.org).

---

## Structure

```
data-science-portfolio/
├── README.md
├── projects/
│   ├── condition-monitoring/
│   │   ├── README.md
│   │   ├── notebooks/
│   │   │   ├── 01_static_load_analysis.ipynb
│   │   │   ├── 02_varying_load_analysis.ipynb
│   │   │   └── 03_learning_based_diagnostics.ipynb
│   │   └── src/
│   │       ├── tsa.py
│   │       ├── sk.py
│   │       ├── envelope_analysis.py
│   │       ├── getrpm_bayesian.py
│   │       ├── cot.py
│   │       └── health_indicators.py
│   └── gearbox-fault-cnn/
│       ├── README.md
│       ├── notebooks/
│       │   ├── 01_data_generation.ipynb
│       │   ├── 02_cnn_training.ipynb
│       │   └── 03_generalisation.ipynb
│       ├── src/
│       │   ├── __init__.py
│       │   ├── lpdm.py
│       │   ├── tvms.py
│       │   ├── preprocessing.py
│       │   ├── cnn.py
│       │   └── simulation.py
│       └── data/
│           └── README.md
└── assets/
    └── cv.pdf
```

---

## What's Next

The generalisation results point to concrete directions:

- **Wider training distribution** — simulate across a range of load conditions and fault severities rather than a single operating point
- **Transfer learning** — fine-tune on small amounts of real-world labelled data after pre-training on synthetic data
- **Digital Twin integration** — use the LPDM as a real-time physics baseline; detect faults as deviations from model predictions rather than classifying from learned distributions
- **Minimum detectable severity** — characterise the threshold at which the CNN can reliably detect faults at varying noise levels
