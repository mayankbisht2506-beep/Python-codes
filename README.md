## 📌 Scientific Objectives

The codes in this repository are designed to:
1.  **Resolve the Hubble Tension:** Demonstrate how a vacuum phase transition at $z \approx 0.65$ bridges the gap between Early (Planck) and Late (SH0ES) measurements.
2.  **Solve the $S_8$ Tension:** Verify that intrinsic vacuum viscosity suppresses structure growth, resolving the clustering anomaly.
3.  **Preserve Early Universe Physics:** Prove that Big Bang Nucleosynthesis (BBN) and CMB observables remain invariant under the modified gravity scaling.
4.  **Validate High-Redshift Predictions:** Confirm consistency with the age of the Universe and JWST-era halo abundances.

---

## 📂 Repository Contents

### 1. 📊 Figure Generation (Paper Reproducibility)
*Scripts that directly generate the figures presented in the manuscript.*

* **`generate_Figure1_lattice_simulation.py`**
    * **Physics:** Monte-Carlo simulation of lattice strain vs. stress. Demonstrates the "Vacuum Exclusion Principle" and lepton stability limits.
    * **Paper Reference:** Section 5, Figure 1.
* **`generate_Figure2_hubble_transition_model.py`**
    * **Physics:** Visualizes the "Vacuum Hardening" phase transition ($z \approx 0.65$) where the Hubble constant relaxes from 67.4 to 73.0 km/s/Mpc.
    * **Paper Reference:** Section 7.1, Figure 2.
* **`generate_Figure3_S8_growth.py`** (Code provided as `validate_growth_suppression.py`)
    * **Physics:** Solves the linear growth equation with a viscous friction term ($\eta \approx 0.17$) to model the suppression of $f\sigma_8$.
    * **Paper Reference:** Section 7.6, Figure 3.
* **`generate_Figure4a_tension.py`**
    * **Physics:** "Tension Metric" test using Pantheon+ Supernovae. Shows resolution of the absolute magnitude discrepancy.
    * **Paper Reference:** Section 8.4, Figure 4 (Top).
* **`generate_Figure4b_validate_shape.py`**
    * **Physics:** "Steel Man" Shape Consistency test. Proves the model's expansion history shape is statistically indistinguishable from optimized $\Lambda$CDM.
    * **Paper Reference:** Section 8.4, Figure 4 (Bottom).
* **`generate_Figure5_cosmic_chronometers_test.py`**
    * **Physics:** Compares $H(z)$ predictions against 31 Cosmic Chronometer measurements.
    * **Paper Reference:** Section 9.3, Figure 5.

### 2. 🧪 Early Universe & BBN Stability
*Scripts proving the model does not break light element abundances.*

* **`validate_helium_stability.py`**
    * **Test:** Verifies the "Cancellation Theorem" for Helium-4 ($Y_p$). Proves that stronger weak interactions exactly offset the faster expansion rate.
    * **Paper Reference:** Section 7.9.
* **`validate_deuterium_robust.py`**
    * **Test:** Checks Deuterium invariance using geometric scaling laws for density and cross-section ($\sigma \propto G$).
    * **Paper Reference:** Section 7.9.3.
* **`validate_lithium_solution.py`**
    * **Test:** Demonstrates how reduced nucleon mass ($m \propto G^{-0.5}$) enhances tunneling, depleting Lithium-7 by $\approx 3x$ to match observations.
    * **Paper Reference:** Section 9.6, Table VI.

### 3. 📉 Structure Growth & Weak Lensing
*Scripts validating the resolution of the Clustering ($S_8$) Tension.*

* **`S8_KiDS_DES.py`**
    * **Test:** Compares the model's predicted $S_8$ against KiDS-1000 and DES-Y3 weak lensing surveys.
    * **Paper Reference:** Section 7.6, Table III.
* **`validate_growth_suppression.py`**
    * **Test:** Validates the growth rate suppression against the "Gold" $f\sigma_8$ dataset (BOSS/WiggleZ).
    * **Paper Reference:** Section 9.5.

### 4. 🔭 Relativistic & Geometric Consistency
*Scripts testing consistency with CMB, BAO, and General Relativity.*

* **`validate_BAO_consistency.py`**
    * **Test:** Checks consistency with "Gold" BAO data (6dFGS, SDSS, BOSS) by accounting for the "Viscous Horizon Contraction" of the sound ruler ($r_s$).
    * **Paper Reference:** Section 9.25, Table VII.
* **`validate_ISW_stability.py`**
    * **Test:** Tests the Integrated Sachs–Wolfe (ISW) effect to ensure late-time potential decay does not distort the CMB power spectrum.
    * **Paper Reference:** Section 9.26.
* **`validate_universe_age.py`**
    * **Test:** Integrates the modified expansion history to ensure the Universe age (>12.5 Gyr) is consistent with globular clusters.
    * **Paper Reference:** Section 9.24.

### 5. 🔭 JWST & High-Redshift Anomalies
* **`validate_JWST_growth.py`**
    * **Test:** Calculates the "Cosmic Turbocharger" enhancement of halo abundance at $z=10$, explaining massive early galaxies.
    * **Paper Reference:** Section 9.24.1.

### 6. 📊 Global Statistics
* **`check_magnitude_shift.py`**
    * **Test:** Diagnostics for high-redshift supernova magnitude residuals ($z > 0.65$). Performs the "Split-Sample Analysis".
    * **Paper Reference:** Section 8.3.

---

## 🛠 Data & Dependencies

**Required Packages:**
```bash
pip install numpy scipy pandas matplotlib requests

Data Sources:
 * Pantheon+ Supernovae: Automatically downloaded from the official GitHub repository.
 * Cosmic Chronometers: Hardcoded compilation from Moresco et al. (31 data points).
 * BAO & f$\sigma_8$: Hardcoded consensus "Gold" datasets (BOSS DR12, 6dFGS, WiggleZ) as defined in the literature.
▶️ Usage Example
Each script is standalone. To run the Hubble Tension visualization:
python generate_Figure2_hubble_transition_model.py

To verify the Lithium Solution:
python validate_lithium_solution.py

📜 License
MIT License. Free for academic use and reproduction with appropriate citation to the main paper.


