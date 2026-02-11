
# Vacuum Elastodynamics: Verification & Validation Suite

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17666785.svg)](https://doi.org/10.5281/zenodo.17666785)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-blue?logo=github)](https://github.com/mayankbisht2506-beep/Python-codes.git)

**Repository for the paper:** *"Vacuum Elastodynamics: Geometric Unification and the Resolution of Hubble and S8 Tensions via Lattice Viscosity"*
```markdown
## 📂 Overview

This repository contains the complete Steel Man validation suite for the Vacuum Elastodynamics model. It consists of 27 independent Python scripts designed to rigorously stress-test the theoretical claims, mathematical derivations, and observational fits presented in the manuscript.

### Resources

- 📄 **Scientific Paper (Zenodo):** https://doi.org/10.5281/zenodo.17666785

- 💻 **Source Code:** Hosted in this GitHub repository.

These scripts demonstrate that the model simultaneously resolves:

- Hubble Tension
- S8 Tension
- Lithium Problem

while preserving ΛCDM successes in:

- Big Bang Nucleosynthesis (BBN)
- Cosmic Microwave Background (CMB)
- Cosmic Age
- Solar System Gravity (Cassini)

---

## 🛠️ Requirements

Install dependencies using:

```bash
pip install numpy scipy pandas matplotlib requests emcee corner uproot awkward camb

```

---

## 🧪 Validation Catalog

### 1. Core Resolution (H₀ & S₈ Tensions)

| Script Name | Objective | Key Result |
| --- | --- | --- |
| Gravity_Boost.py | Verify H₀ shift via Early Gravity Boost | 74.5 km/s/Mpc |
| mcmc_stress.py | Stability stress test | Pass |
| S8_KiDS_DES.py | Verify S₈ suppression | 0.776 |
| generate_Figure3a_tension.py | Hubble tension significance | > 5σ |
| validate_vacuum_tension_resolution.py | Pantheon+ stress test | Δχ² = -4973.1 |

---

### 2. Microphysics & Fundamental Constants

| Script Name | Objective | Key Result |
| --- | --- | --- |
| generate_Figure1_lattice_simulation.py | Lepton sum rule simulation | 98.6% |
| validate_lithium_solution.py | Lithium-7 depletion | 2.76× |
| validate_deuterium_robust.py | D/He invariance | < 1.0% |
| validate_BBN_stability.py | Helium-4 stability | Invariant |
| generate_Figure5_validate_vacuum_fracture.py | Vacuum fracture test | 0.9 TeV |

---

### 3. Consistency Checks (Safety Mechanisms)

| Script Name | Objective | Key Result |
| --- | --- | --- |
| verify_CMB_geometric_scaling.py | CMB Spectrum Restoration (CAMB) | Peaks Aligned |
| hyperuniform_screening_check.py | Cassini (Solar System) Check | Range < 12mm |
| validate_CMB_invariance.py | Acoustic Scale Stability | 0.26% |
| validate_jerk_stability.py | Singularity check | Jerk < 3.59 |
| Universe_age.py | Universe age check | 13.05 Gyr |
| validate_ISW_stability.py | ISW stability | ×1.10 |
| generate_Figure3b_validate_shape.py | Distance shape | Δχ² ≈ 0.28 |
| hubble_transition_model.py | Phase transition | Smooth |

---

### 4. Observational Probes (BAO, Growth, Galaxies)

| Script Name | Objective | Key Result |
| --- | --- | --- |
| validate_Pk_screening.py | Power spectrum screening | 0.00% |
| validate_BAO_consistency.py | BAO ruler test | 0.905× |
| validate_BAO_ladder.py | Distance ladder | < 1σ |
| validate_growth_numerical.py | Growth rate | S₈ ≈ 0.776 |
| check_magnitude_shift.py | SNe shift | < 1σ |
| validate_JWST_growth.py | Galaxy growth | ×2458 |
| cosmic_chronometers_test.py | H(z) test | χ² ≈ 0.79 |

---

### 5. Global Statistical Verdict

| Script Name | Objective | Result |
| --- | --- | --- |
| validate_global_stats.py | Global likelihood sum | Δχ² ≈ -3568 |
| generate_Figure4_MCMC_validation.py | Bayesian check | Pass |

---

## 📊 Key Findings

Running:

```bash
python src/validate_global_stats.py

```

reproduces the main conclusion:

> **The Unified Vacuum Model is globally preferred by >5σ over ΛCDM.**

### Summary

* Supernovae: H₀ ≈ 74.5
* Structure Growth: S₈ ≈ 0.776
* Matter Power: Preserved
* BBN: Li7 resolved, D/He preserved
* CMB: Acoustic scale preserved (Peaks Aligned via CAMB)
* Solar System: Fifth Force screened (< 12mm)
* Vacuum Fracture: 0.9 TeV detection

---

## 📁 Repository Structure

```
/src/       # Validation scripts
/figures/   # Generated plots
/data/      # Cached datasets

```

---

## 📝 Usage

### Global Statistics (Table VIII)

```bash
python src/validate_global_stats.py


```

---

## 📜 Citation

If you use this repository, please cite:

```
Bisht, M. (2026).
Vacuum Elastodynamics: Geometric Unification and the Resolution
of Hubble and S8 Tensions via Lattice Viscosity.
Zenodo. [https://doi.org/10.5281/zenodo.17666785](https://doi.org/10.5281/zenodo.17666785)

```

```

```
