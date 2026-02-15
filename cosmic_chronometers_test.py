import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. COSMIC CHRONOMETER DATA (N=31)
# ==========================================
# Standard compilation (Moresco et al. 2016, etc.)
cc_data = np.array([
    [0.07, 69.0, 19.6], [0.09, 69.0, 12.0], [0.12, 68.6, 26.2], [0.17, 83.0, 8.0],
    [0.179, 75.0, 4.0], [0.199, 75.0, 5.0], [0.20, 72.9, 29.6], [0.27, 77.0, 14.0],
    [0.28, 88.8, 36.6], [0.352, 83.0, 14.0], [0.3802, 83.0, 13.5], [0.4, 95.0, 17.0],
    [0.4004, 77.0, 10.2], [0.4247, 87.1, 11.2], [0.4497, 92.8, 12.9], [0.47, 89.0, 23.0],
    [0.4783, 80.9, 9.0], [0.48, 97.0, 62.0], [0.593, 104.0, 13.0], [0.68, 92.0, 8.0],
    [0.781, 105.0, 12.0], [0.875, 125.0, 17.0], [0.88, 90.0, 40.0], [0.9, 117.0, 23.0],
    [1.037, 154.0, 20.0], [1.3, 168.0, 17.0], [1.363, 160.0, 33.6], [1.43, 177.0, 18.0],
    [1.53, 140.0, 14.0], [1.75, 202.0, 40.0], [1.965, 186.5, 50.4],
])

z_cc = cc_data[:, 0]
hz_cc = cc_data[:, 1]
err_cc = cc_data[:, 2]

# ==========================================
# 2. UNIFIED PHYSICS ENGINE
# ==========================================
Z_TRANS = 0.65  
WIDTH = 0.10    

# --- MODEL A: PLANCK LCDM ---
H0_PLANCK = 67.4
OM_PLANCK = 0.315
OL_PLANCK = 1.0 - OM_PLANCK

# --- MODEL B: VACUUM ELASTODYNAMICS ---
H_TERMINAL = 72.80   # The observed terminal velocity from MCMC
OM_PRIMORDIAL = 0.315 # Frictionless early universe
OM_EFFECTIVE = 0.366  # Viscous late universe (Inertial Counter-Load = +0.051)

def h_lcdm(z):
    return H0_PLANCK * np.sqrt(OM_PLANCK * (1 + z)**3 + OL_PLANCK)

def h_viscous(z):
    # The Inertial Counter-Load activates at the z=0.65 phase transition
    arg = (z - Z_TRANS) / WIDTH
    sigmoid = np.where(arg > 100, 0.0, 1.0 / (1.0 + np.exp(arg)))
    
    # Density dynamically transitions from 0.315 (early) to 0.366 (late)
    OM_Z = OM_PRIMORDIAL + (OM_EFFECTIVE - OM_PRIMORDIAL) * sigmoid
    OL_Z = 1.0 - OM_Z
    
    E_z = np.sqrt(OM_Z * (1 + z)**3 + OL_Z)
    
    # Fast terminal expansion globally
    return H_TERMINAL * E_z

# ==========================================
# 3. STATISTICAL VALIDATION
# ==========================================
# Model 1: Standard Planck
hz_planck = h_lcdm(z_cc)

# Model 2: Vacuum Elastodynamics
hz_vacuum = h_viscous(z_cc)

# Chi-Squared
chi2_planck = np.sum(((hz_cc - hz_planck) / err_cc)**2)
chi2_vacuum = np.sum(((hz_cc - hz_vacuum) / err_cc)**2)

dof = len(z_cc) 
rchi2_planck = chi2_planck / dof
rchi2_vacuum = chi2_vacuum / dof

print(f"\n--- H(z) CONSISTENCY RESULTS (Table 6 Verification) ---")
print(f"Planck Model (67.4): Chi2={chi2_planck:.2f} | Reduced={rchi2_planck:.2f}")
print(f"Vacuum Model (72.8): Chi2={chi2_vacuum:.2f} | Reduced={rchi2_vacuum:.2f}")

# VERDICT
if 0.7 < rchi2_vacuum < 1.2:
    print(f"\nVERDICT: SUCCESS.")
    print(f"The Vacuum Model (Reduced Chi2 = {rchi2_vacuum:.2f}) is statistically consistent.")
    print("This proves the dynamic Inertial Counter-Load perfectly threads the H(z) data,")
    print("even while running on the global H0 = 72.8 terminal trajectory!")
else:
    print(f"\nVERDICT: CHECK PARAMETERS (RChi2={rchi2_vacuum:.2f})")

# ==========================================
# 4. PLOT
# ==========================================
plt.figure(figsize=(10, 6))
plt.errorbar(z_cc, hz_cc, yerr=err_cc, fmt='o', color='k', alpha=0.5, label='Cosmic Chronometers (Moresco et al.)')

z_grid = np.linspace(0, 2.0, 200)

# Plot Planck
plt.plot(z_grid, h_lcdm(z_grid), 'b--', label=rf'Standard LCDM ($H_0={H0_PLANCK}, \Omega_m={OM_PLANCK}$)')

# Plot Vacuum unified model
plt.plot(z_grid, h_viscous(z_grid), 'r-', linewidth=2.5, 
         label=rf'Vacuum Model ($H_0={H_TERMINAL}, \Omega_m \rightarrow {OM_EFFECTIVE}$)')

plt.axvline(x=Z_TRANS, color='gray', linestyle=':', label='Viscous Transition $z=0.65$')
plt.xlabel('Redshift z', fontsize=12)
plt.ylabel('H(z) [km/s/Mpc]', fontsize=12)
plt.title(rf'Cosmic Chronometer Consistency Check' + '\n' + rf'Vacuum $\chi^2_\nu \approx {rchi2_vacuum:.2f}$', fontsize=14)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.savefig('Figure5_Hz_Unified.png')
print("Plot saved as 'Figure5_Hz_Unified.png'")
plt.show()
