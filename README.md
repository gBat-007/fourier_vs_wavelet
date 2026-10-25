# Fourier vs Wavelet

This project studies the time-frequency concentration of Fourier and wavelet transforms for Gaussian pulses and linear chirps.

The goal is to compare how each method localizes signal energy in time and frequency and to analyze how that concentration changes depending on the signal.

## Repository Contents

- `developing_figures.py` — generates the main figures comparing Fourier and wavelet representations.
- `variance_calculations.py` — computes the numerical uncertainty and concentration metrics.
- `main.pdf` — paper/report associated with the project.

## Running the Code

### Requirements

Install the required Python packages:

```bash
pip install numpy scipy matplotlib PyWavelets
```

### Generate figures

```bash
python developing_figures.py
```

This script produces the time-frequency visualizations for the Gaussian pulse and chirp examples.

### Compute metrics

```bash
python variance_calculations.py
```

This script calculates the quantitative concentration/uncertainty results used in the analysis.
