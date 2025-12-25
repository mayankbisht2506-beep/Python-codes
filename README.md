# Vacuum Elastodynamics: Code Repository

This repository contains the Python scripts and data analysis tools supporting the manuscript **"Vacuum Elastodynamics: Resolving the Hubble and S8 Tensions via Lattice Viscosity and Phantom Dark Energy"** by Mayank Bisht.

The code provided here reproduces the theoretical models, numerical simulations, and observational tests presented in Figures 1–5 of the paper.

## 📂 Repository Contents

### 1. Lattice Mechanics & Stability (Section 6)
* **`Figure1_Lattice_Simulation.py`**
  * **Physics:** Performs a Monte Carlo simulation of the $E_8$ vacuum lattice strain fluctuations.
  * **Key Result:** Determines the Frenkel Yield Strength ($\gamma_{crit} \approx 0.159$) and verifies that the sum of the three observed lepton generations saturates 98.6% of this limit, while a 4th generation would be mechanically unstable.

### 2. The Hubble Tension (Section 7 & 8)
* **`Figure2_Hubble_Transition.py`**
  * **Physics:** Visualizes the theoretical phase transition of the vacuum stiffness.
  * **Key Result:** Illustrates the sigmoidal relaxation of $H_0$ from the Planck value ($z>0.65$) to the SH0ES value ($z<0.65$).

* **`Figure4a_Pantheon_Tension.py`**
  * **Physics:** Tests the "Tension Metric" using the Pantheon+ dataset (Absolute Magnitude).
  * **Key Result:** Demonstrates that the "Magnitude Mirage" (Luminosity Drift) resolves the discrepancy between Planck and SH0ES ($\Delta \chi^2 \approx -376$).

* **`Figure4b_Pantheon_Shape.py`**
  * **Physics:** Tests the "Shape Metric" (Steel Man Test) by marginalizing over absolute calibration.
  * **Key Result:** Proves that the "kinked" expansion history fits the shape of the supernova data slightly better than $\Lambda$CDM ($\Delta \chi^2 \approx -0.57$).

* **`check_magnitude_shift.py`**
  * **Physics:** A supplementary validation script that calculates the raw mean residual of high-redshift supernovae ($z > 0.65$) relative to the Planck baseline.
  * **Key Result:** Empirically verifies the $\approx -0.20$ mag shift cited in the text, confirming the "Mirage" effect is present in the raw data.

* **`Figure5_Cosmic_Chronometers.py`**
  * **Physics:** Tests the model's $H(z)$ predictions against 31 independent cosmic chronometer measurements.
  * **Key Result:** Confirms that the "Late-Time Boost" model is statistically indistinguishable from the observed expansion history ($\Delta AIC < 2$).

* **`BAO_DATA.py`**
  * **Physics:** Validates the model against Baryon Acoustic Oscillation (BAO) measurements ($D_V/r_d$).
  * **Key Result:** Confirms consistency with 6dFGS, SDSS MGS, and BOSS DR12 data.

### 3. The S8 Tension (Section 9)
* **`Figure3_S8_Resolution.py`**
  * **Physics:** Solves the differential equations for linear structure growth (`odeint`), incorporating Lattice Viscosity ($\eta \approx 0.17$).
  * **Key Result:** Demonstrates that vacuum viscosity dynamically suppresses the growth rate ($f\sigma_8$), resolving the clustering tension ($\Delta \chi^2 = -1.81$) without modifying General Relativity.

---

## 🛠️ Requirements

To run these scripts, you will need a standard Python 3 environment with the following libraries:

* `numpy`
* `matplotlib`
* `pandas`
* `scipy`
* `requests` (for downloading Pantheon+ data)

You can install all dependencies via pip:
```bash
pip install numpy matplotlib pandas scipy requests


