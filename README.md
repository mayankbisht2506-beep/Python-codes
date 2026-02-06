# Vacuum Elastodynamics: Verification & Validation Suite

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17666785.svg)](https://doi.org/10.5281/zenodo.17666785)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-blue?logo=github)](https://github.com/mayankbisht2506-beep/Vacuum-Elastodynamics-Validation)

**Repository for the paper:** *"Vacuum Elastodynamics: Geometric Unification and the Resolution of Hubble and S8 Tensions via Lattice Viscosity"*

## 📂 Overview
This repository contains the complete "Steel Man" validation suite for the Vacuum Elastodynamics model. It consists of **25 independent Python scripts** designed to strictly stress-test the theoretical claims, mathematical derivations, and observational fits presented in the manuscript.

* **📄 Scientific Paper:** The full manuscript is archived and citable via Zenodo: **[https://doi.org/10.5281/zenodo.17666785](https://doi.org/10.5281/zenodo.17666785)**
* **💻 Source Code:** All validation scripts and data processing pipelines are hosted in this GitHub repository.

These scripts demonstrate that the model simultaneously resolves the **Hubble Tension**, **S8 Tension**, and **Lithium Problem** while preserving the successes of the Standard Model ($\Lambda$CDM) in BBN, CMB, and Cosmic Age.

## 🛠️ Requirements
To run these scripts, you will need a standard Python 3 scientific stack plus MCMC tools:
```bash
pip install numpy scipy pandas matplotlib requests emcee corner uproot awkward

🧪 Validation Catalog
1. The Core Resolution (H_0 & S_8 Tensions)
Tests verifying the mechanism's ability to fix the primary cosmological crises.
| Script Name | Objective | Key Result (Matches Paper) |
|---|---|---|
| Gravity_Boost.py | Verify H_0 shift via Early Gravity Boost (G_{early}). | 74.5 km/s/Mpc (Matches SH0ES) |
| mcmc_stress.py | Stability Stress Test fixing H_0 & S_8. | Pass (Matches Geometric Ideal) |
| S8_KiDS_DES.py | Verify S_8 suppression via Vacuum Viscosity (\nu). | 0.776 (Matches KiDS/DES within 0.3σ) |
| generate_Figure2a_tension.py | Calculate significance for the Hubble Tension resolution. | > 5$\sigma$ Significance |
| validate_vacuum_tension_resolution.py | Test I: Raw stress test against Pantheon+ SNe data. | Raw \Delta\chi^2: -4973.1 (Massive Headroom) |
2. Microphysics & Fundamental Constants
Tests validating the geometric origin of mass, the fracture limits, and the solution to the Lithium Problem.
| Script Name | Objective | Key Result (Matches Paper) |
|---|---|---|
| generate_Figure1_lattice_simulation.py | Simulate Lepton Sum Rule & Frenkel Limit saturation. | Saturation: 98.6% (Explains 3 Generations) |
| validate_lithium_solution.py | Verify Lithium-7 depletion via Tunneling Barrier reduction. | Depletion: 2.76x (Solves Li Problem) |
| validate_deuterium_robust.py | Verify Deuterium/Helium invariance ("Cancellation Theorem"). | Drift: < 1.0% (Symmetry Preserved) |
| validate_BBN_stability.py | Full BBN stability check for Helium-4 (Y_p). | Invariant (Matches Planck) |
| validate_vacuum_fracture.py | NEW: Verify Vacuum Fracture (CMS Control vs DAMPE Signal). | Break @ 0.9 TeV (Matches 1.0 TeV Prediction) |
3. Consistency Checks (Safety Mechanisms)
Tests ensuring the model does not break established physics (CMB, Age, Shapes, Kinematics).
| Script Name | Objective | Key Result (Matches Paper) |
|---|---|---|
| validate_jerk_stability.py | Check for "Cosmic Whiplash" (singularities). | Max Jerk < 3.59 (Adiabatically Smooth) |
| validate_CMB_invariance.py | Check stability of CMB Acoustic Scale (\theta_*). | Error 0.26% (Preserves Planck Fit) |
| Universe_age.py | Ensure H_0 does not violate Globular Cluster ages. | Age = 13.05 Gyr (Pass > 12.5 Gyr) |
| validate_ISW_stability.py | Check Integrated Sachs-Wolfe (ISW) stability. | Power x1.10 (Consistent w/ Variance) |
| generate_Figure2b_validate_shape.py | Verify Luminosity Distance shape consistency. | \Delta \chi^2 \approx 0.28 (Indistinguishable) |
| hubble_transition_model.py | Validate the phase transition profile at z \approx 0.65. | Success (Smooth Transition) |
4. Observational Probes (BAO, Growth, Galaxies)
Tests against specific dataset constraints.
| Script Name | Objective | Key Result (Matches Paper) |
|---|---|---|
| validate_Pk_screening.py | Verify Environmental Screening preserves P(k) shape. | Turnover Shift: 0.00% (Matches SDSS Shape) |
| validate_BAO_consistency.py | Check BAO standard ruler contraction (r_d). | Tie (Corrects ruler by 0.905x) |
| validate_BAO_ladder.py | Verify Inverse Distance Ladder residuals. | Residuals < 1$\sigma$ |
| validate_growth_numerical.py | Verify Growth Rate (f\sigma_8) evolution. | Global Tie (S_8 \approx 0.776) |
| check_magnitude_shift.py | Check SNe Magnitude Shift in Deep Field. | Z-Score < 1$\sigma$ (Excellent Match) |
| validate_JWST_growth.py | Calculate Halo Mass enhancement for JWST galaxies. | Boost ~2458x (Solves "Impossible Galaxies") |
| cosmic_chronometers_test.py | Consistency check with Cosmic Chronometers (H(z)). | \chi_\nu^2 \approx 0.79 (Consistent) |
5. Global Statistical Verdict
The final audit of the model's performance across all datasets.
| Script Name | Objective | Global Net Evidence |
|---|---|---|
| validate_global_stats.py | Sum of all statistical likelihoods (Table VIII). | \Delta\chi^2 \approx -3568.0 |
| MCMC_validation.py | Bayesian Parameter Estimation prediction check. | Pass (Consistent with Theory) |
📊 Key Findings
Running validate_global_stats.py reproduces the paper's main conclusion:
> The Unified Vacuum Model is globally preferred by >5\sigma over \LambdaCDM.
> 
 * Supernovae (Pantheon+): Decisive Resolution (H_0 \approx 74.5)
 * Structure Growth: Statistical Tie (S_8 \approx 0.776)
 * Matter Power Spectrum: Shape Preserved (via Screening)
 * BBN: Perfect Invariance (Resolves Li7, preserves D/He)
 * CMB: Geometric Resonance (Preserves Acoustic Scale)
 * Vacuum Fracture: Detected at 0.9 TeV (Matches Theory 1.0 TeV)
📁 Repository Structure
 * /src/: Contains all 25 validation scripts listed above.
 * /figures/: Outputs from plotting scripts (e.g., vacuum_fracture_test_corrected.png).
 * /data/: Local cache for downloaded Pantheon+ data.
📝 Usage
To verify the Global Statistical Budget (Table VIII in the paper):
python src/validate_global_stats.py

To run the Vacuum Fracture Test (verifying the 1 TeV Limit):
python src/validate_vacuum_fracture.py

📜 Citation
If you use these scripts or the Vacuum Elastodynamics model in your work, please cite the original manuscript:
> Bisht, M. (2026). Vacuum Elastodynamics: Geometric Unification and the Resolution of Hubble and S8 Tensions via Lattice Viscosity. Zenodo. https://doi.org/10.5281/zenodo.17666785
> 
