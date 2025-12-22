# Vacuum Elastodynamics: Code Repository

This repository contains the Python scripts and data analysis tools supporting the manuscript **"Vacuum Elastodynamics: Resolving the Hubble and S8 Tensions via Lattice Viscosity and Phantom Dark Energy"** by Mayank Bisht.

The code provided here reproduces the theoretical models, numerical simulations, and observational tests presented in Figures 1–5 of the paper.

## Repository Contents

### 1. Lattice Mechanics & Stability (Section 6)
* **`lattice_simulation.py` (Corresponds to Figure 1 & Table 1)**
    * **Physics:** Performs a Monte Carlo simulation of the $E_8$ vacuum lattice strain fluctuations.
    * **Key Result:** Determines the Frenkel Yield Strength ($\gamma_{crit} \approx 0.159$) and verifies that the sum of the three observed lepton generations saturates 98.6% of this limit, while a 4th generation would be mechanically unstable.

### 2. The Hubble Tension (Section 7 & 8)
* **`hubble_transition_model.py` (Corresponds to Figure 2)**
    * **Physics:** Visualizes the theoretical phase transition of the vacuum stiffness.
    * **Key Result:** Illustrates the sigmoidal relaxation of $H_0$ from the Planck value ($z>0.65$) to the SH0ES value ($z<0.65$).

* **`pantheon_validation.py` (Corresponds to Figure 4)**
    * **Physics:** Analyzes Type Ia Supernovae magnitude residuals from the Pantheon+ dataset.
    * **Key Result:** Identifies the magnitude shift $\Delta \mu \approx -0.2$ at $z > 0.65$, providing observational evidence for vacuum relaxation.

* **`cosmic_chronometers_test.py` (Corresponds to Figure 5 & Table 4)**
    * **Physics:** Tests the model's $H(z)$ predictions against 31 independent cosmic chronometer measurements.
    * **Key Result:** Confirms that the "Late-Time Boost" model is statistically indistinguishable from the observed expansion history ($\Delta AIC < 2$).

* **`check_magnitude_shift.py` (Supplementary Check)**
    * **Physics:** Calculates the raw mean residual of high-redshift supernovae ($z > 0.65$) relative to the Planck baseline.
    * **Key Result:** Empirically verifies the $-0.20$ mag shift cited in the text.

### 3. The S8 Tension (Section 9)
* **`S8_viscous_solver.py` (Corresponds to Figure 3)**
    * **Physics:** Solves the differential equations for linear structure growth, including a term for Lattice Viscosity ($\eta \approx 0.17$).
    * **Key Result:** Demonstrates that vacuum viscosity dynamically suppresses the growth rate ($f\sigma_8$), resolving the clustering tension ($\Delta \chi^2 < 0$) without modifying General Relativity.

## Requirements
To run these scripts, you will need a standard Python 3 environment with the following libraries:

* `numpy`
* `matplotlib`
* `pandas`
* `scipy`
* `requests` (for downloading Pantheon+ data)
* `astropy` (for precise cosmological calculations)

You can install the dependencies via pip:
```bash
pip install numpy matplotlib pandas scipy requests astropy

Bisht, M. (2025). "Vacuum Elastodynamics: Resolving the Hubble and S8 Tensions via Lattice Viscosity and Phantom Dark Energy." Physica Scripta (Submitted). DOI: 10.5281/zenodo.18014968
