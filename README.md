# Vacuum Elastodynamics: Verification & Validation Suite

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17666785.svg)](https://doi.org/10.5281/zenodo.17666785)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-blue?logo=github)](https://github.com/mayankbisht2506-beep/Python-codes.git)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
```markdown
**Repository for the paper:** *"Vacuum Elastodynamics: Geometric Unification and the Resolution of Hubble and S8 Tensions via Lattice Viscosity"*

## 📂 Overview

This repository contains the complete **"Steel Man" Validation Suite** for the Vacuum Elastodynamics (VED) framework. It consists of **24 independent Python scripts** designed to rigorously stress-test the theoretical claims, mathematical derivations, and observational fits presented in the manuscript.

### Resources

- 📄 **Scientific Paper (Zenodo):** [https://doi.org/10.5281/zenodo.17666785](https://doi.org/10.5281/zenodo.17666785)
- 💻 **Source Code:** Hosted in this GitHub repository.

These scripts demonstrate that the model simultaneously resolves:
- **Hubble Tension** ($H_0 \approx 72.87$)
- **$S_8$ Tension** ($S_8 \approx 0.767$)
- **Lithium Problem** (2.63x Depletion)

...while preserving $\Lambda$CDM successes in BBN, CMB, and Solar System gravity.

---

## 🛠️ Requirements

Install dependencies using:

```bash
pip install numpy scipy pandas matplotlib requests emcee corner uproot awkward camb
```

---

## 🧪 Validation Catalog

### 1. Core Resolution (H₀ & S₈ Tensions)
*Verifying the mechanical resolution of the expansion and growth tensions.*

| Script Name | Objective | Key Result (Paper) |
| :--- | :--- | :--- |
| `Gravity_Boost.py` | Verify H₀ shift via Early Gravity Boost | $H_{fast} \approx 74.69$ |
| `S8_KiDS_DES.py` | Verify S₈ suppression via Viscosity | $S_8 \approx 0.767$ |
| `generate_Figure3a_tension.py` | Hubble tension significance check | $\Delta\chi^2 \approx -2475.72$ |
| `validate_vacuum_tension_resolution.py` | Pantheon+ Stress Test (Raw $\chi^2$) | $\Delta\chi^2 \approx -2492.58$ |

### 2. Microphysics & Fundamental Constants
*Verifying the geometric derivation of constants and particle stability.*

| Script Name | Objective | Key Result (Paper) |
| :--- | :--- | :--- |
| `generate_Figure1_lattice_simulation.py` | Lepton Saturation Sum Rule | $98.58\%$ Efficiency |
| `validate_lithium_solution.py` | Lithium-7 Depletion ($m \propto G^{-0.5}$) | $2.63\times$ Depletion |
| `validate_deuterium_robust.py` | Deuterium/Helium Invariance | Invariant |
| `validate_BBN_stability.py` | Helium-4 Freeze-out Check | $Y_p \Invariant |
| `generate_Figure5_validate_vacuum_fracture.py` | Vacuum Fracture (Electron/Proton) | $0.9$ TeV |

### 3. Consistency Checks (Safety Mechanisms)
*Ensuring the new physics does not break existing precision observations.*

| Script Name | Objective | Key Result (Paper) |
| :--- | :--- | :--- |
| `verify_CMB_geometric_scaling.py` | CMB Spectrum Restoration (CAMB) | Peaks Aligned |
| `hyperuniform_screening_check.py` | Cassini (Solar System) Screening | Range $< 100$m |
| `validate_CMB_invariance.py` | Acoustic Scale Stability ($\theta_*$) | Error $< 0.01\%$ |
| `validate_jerk_stability.py` | Kinematic Singularity Check | $j_{max} \approx 1.3$ |
| `Universe_age.py` | Cosmic Age Calculation | $12.52$ Gyr |
| `validate_ISW_stability.py` | Integrated Sachs-Wolfe (Supervoids) | Signal $\times 1.22$ |
| `generate_Figure3b_validate_shape.py` | Expansion History Shape Test | $\Delta\chi^2 \approx +9.23$ |
| `hubble_transition_model.py` | Phase Transition Smoothness | Smooth |

### 4. Observational Probes (BAO, Growth, Galaxies)
*Cross-validating against independent datasets.*

| Script Name | Objective | Key Result (Paper) |
| :--- | :--- | :--- |
| `validate_Pk_screening.py` | Matter Power Spectrum Shape | $0.00\%$ Deviation |
| `validate_BAO_ladder.py` | BAO Distance Ladder (8-Point) | $∆χ2≈ 1.61$ |
| `validate_growth_numerical.py` | Linear Growth Rate ($f\sigma_8$) | $∆χ2 ≈ −1.11$ |
| `validate_JWST_growth.py` | JWST "Impossible Galaxies" | Luminosity Boost |
| `cosmic_chronometers_test.py` | Cosmic Chronometers ($H(z)$) | $\chi^2_\nu \approx 0.95$ |

### 5. Global Statistical Verdict
*The final Bayesian evidence summary.*

| Script Name | Objective | Result |
| :--- | :--- | :--- |
| `validate_global_stats.py` | Global Likelihood Sum (Net) | $\Delta\chi^2 \approx -2461.80$ |
| `generate_Figure4_MCMC_validation.py` | Blind MCMC Parameter Recovery | $H_0 = 72.36 \pm 0.24$ |

---

## 📊 Key Findings

Running `src/validate_global_stats.py` reproduces the main conclusion:

> **The Unified Vacuum Model is globally preferred by >5σ over ΛCDM.**

### Summary Scorecard
*   **Supernovae:** $H_0 \approx 72.36$ km/s/Mpc (Matches SH0ES)
*   **Structure Growth:** $S_8 \approx 0.767$ (Matches Weak Lensing)
*   **BBN:** Lithium-7 Solved, D/He Preserved
*   **CMB:** Acoustic Scale Locked ($\theta_*$ invariant)
*   **Solar System:** Fifth Force Screened (Hyperuniformity)
*   **Vacuum Fracture:** 0.9 TeV Break Detected (DAMPE/CMS)

---

## 📁 Repository Structure

```
/src/       # Validation scripts (25 files)
/figures/   # Generated plots (Figures 1-5)
/data/      # Cached datasets (Pantheon+, Chronometers)
```

---

## 📝 Usage

To run the **"Acid Test"** (Blind MCMC) to verify the expansion history:

```bash
python src/generate_Figure4_MCMC_validation.py
```

To verify the **Lithium-7 Solution**:

```bash
python src/validate_lithium_solution.py
```

---

## 📜 Citation

If you use this repository, please cite:

```bibtex
@article{Bisht2026,
  title={Vacuum Elastodynamics: Geometric Unification and the Resolution of Hubble and S8 Tensions via Lattice Viscosity},
  author={Bisht, Mayank},
  journal={Zenodo},
  year={2026},
  doi={10.5281/zenodo.17666785}
}
```
```
