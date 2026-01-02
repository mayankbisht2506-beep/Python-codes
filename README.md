```markdown
# Vacuum Elastodynamics: Verification & Validation Suite

**Repository for the paper:** *"Vacuum Elastodynamics: Resolving the Hubble and S8 Tensions via Lattice
Viscosity"*

## 📂 Overview
This repository contains the complete "Steel Man" validation suite for the Vacuum Elastodynamics model. It consists of **21 independent Python scripts** designed to strictly stress-test the theoretical claims, mathematical derivations, and observational fits presented in the manuscript.

These scripts demonstrate that the model simultaneously resolves the **Hubble Tension**, **S8 Tension**, and **Lithium Problem** while preserving the successes of the Standard Model ($\Lambda$CDM) in BBN, CMB, and Cosmic Age.

## 🛠️ Requirements
To run these scripts, you will need a standard Python 3 scientific stack:
```bash
pip install numpy scipy pandas matplotlib requests

```

---

## 🧪 Validation Catalog

### 1. The Core Resolution ( &  Tensions)

Tests verifying the mechanism's ability to fix the primary cosmological crises.

| Script Name | Objective | Key Result (Matches Paper) |
| --- | --- | --- |
| `Gravity_Boost.py` | Verify  shift via Early Gravity Boost (). | **** (Matches SH0ES) |
| `S8_KiDS_DES.py` | Verify  suppression via Vacuum Viscosity (). | **** (Matches KiDS/DES) |
| `generate_Figure2a_tension.py` | Calculate  for the Hubble Tension resolution. | **** |
| `validate_vacuum_tension_resolution.py` | Raw stress test against Pantheon+ SNe data. | **Raw ** (Pass) |

### 2. Microphysics & Fundamental Constants

Tests validating the geometric origin of mass and the solution to the Lithium Problem.

| Script Name | Objective | Key Result (Matches Paper) |
| --- | --- | --- |
| `generate_Figure1_lattice_simulation.py` | Simulate Lepton Sum Rule & Frenkel Limit saturation. | **Saturation: 98.6%** (Explains 3 Generations) |
| `validate_lithium_solution.py` | Verify Lithium-7 depletion via Tunneling Barrier reduction. | **Depletion: 3.39x** (Solves Li Problem) |
| `validate_deuterium_robust.py` | Verify Deuterium/Helium invariance ("Cancellation Theorem"). | **Drift: 0.00%** (Perfect Symmetry) |
| `validate_BBN_stability.py` | Full BBN stability check for Helium-4 (). | **Invariant** (Matches Planck) |

### 3. Consistency Checks (Safety Mechanisms)

Tests ensuring the model does not break established physics (CMB, Age, Shapes).

| Script Name | Objective | Key Result (Matches Paper) |
| --- | --- | --- |
| `validate_CMB_invariance.py` | Check stability of CMB Acoustic Scale (). | **Error < 0.7%** (Preserves Planck Fit) |
| `Universe_age.py` | Ensure  does not violate Globular Cluster ages. | **Age = 13.21 Gyr** (Pass > 12.5 Gyr) |
| `validate_ISW_stability.py` | Check Integrated Sachs-Wolfe (ISW) stability. | **Boost ** (Pass) |
| `generate_Figure2b_validate_shape.py` | Verify Luminosity Distance shape consistency. | **** |
| `hubble_transition_model.py` | Validate the phase transition profile at . | **Success** (Smooth Transition) |

### 4. Observational Probes (BAO, Growth, Galaxies)

Tests against specific dataset constraints.

| Script Name | Objective | Key Result (Matches Paper) |
| --- | --- | --- |
| `validate_BAO_consistency.py` | Check BAO standard ruler contraction (). | **Tie** (Corrects ruler by 0.924x) |
| `validate_BAO_ladder.py` | Verify Inverse Distance Ladder residuals. | **Residuals < 0.10** |
| `validate_growth_numerical.py` | Verify Growth Rate () evolution. | **Global Tie** () |
| `check_magnitude_shift.py` | Check SNe Magnitude Shift in Deep Field. | **Z-Score < 1$\sigma$** (Excellent Match) |
| `validate_JWST_growth.py` | Calculate Halo Mass enhancement for JWST galaxies. | **Boost ** (Solves "Impossible Galaxies") |
| `cosmic_chronometers_test.py` | Consistency check with Cosmic Chronometers . | **** (Consistent) |

### 5. Global Statistical Verdict

The final audit of the model's performance across all datasets.

| Script Name | Objective | Global Net Evidence |
| --- | --- | --- |
| `validate_global_stats.py` | Sum of all statistical likelihoods (Table VIII). | **** |
| `MCMC_validation.py` | Bayesian Parameter Estimation prediction check. | **** (Consistent with Theory) |

---

## 📊 Key Findings

Running `validate_global_stats.py` reproduces the paper's main conclusion:

> **The Unified Vacuum Model is globally preferred by  over CDM.**

* **Supernovae (Pantheon+):** Decisive Resolution ()
* **Structure Growth:** Statistical Tie (Resolves S8 Tension)
* **BBN:** Perfect Invariance (Resolves Li7, preserves D/He)
* **CMB:** Geometric Resonance (Preserves Acoustic Scale)

## 📁 Repository Structure

* `/src/`: Contains all 21 validation scripts listed above.
* `/figures/`: Outputs from plotting scripts (e.g., `Figure1_Stress_Strain.png`).
* `/data/`: (Optional) Local cache for downloaded Pantheon+ data.

## 📝 Usage

To verify the **Global Statistical Budget** (Table VIII in the paper):

```bash
python src/validate_global_stats.py

```

To verify the **Lithium Solution**:

```bash
python src/validate_lithium_solution.py

```

To verify the **Hubble Tension Resolution**:

```bash
python src/Gravity_Boost.py

```

---

*This codebase serves as the "Steel Man" verification for the manuscript "Vacuum Elastodynamics: Resolving the Hubble and S8 Tensions via Lattice
Viscosity". All physics parameters used here are derived strictly from the theoretical constraints presented in the text.*

```

```
