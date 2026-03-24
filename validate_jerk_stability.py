# Uncomment the line below if running in Google Colab / Jupyter
# !pip install scipy numpy matplotlib

import numpy as np
import matplotlib.pyplot as plt

print("--- COSMIC WHIPLASH (JERK PARAMETER STABILITY) ---")
print("Objective: Verify the Phase Transition is Adiabatically Smooth.")
print("Engine: Continuous Parameter Interpolation (Matches Section 8.3.1)")

# ==========================================
# 1. PHYSICS PARAMETERS (High-Precision VED)
# ==========================================
Z_TRANS = 0.641
WIDTH = 0.10

# EARLY REGIME (z > 0.641) - Superfluid Epoch
H_EARLY = 74.69
OM_EARLY = 0.3116

# LATE REGIME (z < 0.641) - Viscous Epoch
H_LATE = 72.71
OM_LATE = 0.3639  # Includes Macroscopic Inertial Counter-Load

# ==========================================
# 2. RIGOROUS DUAL-TRANSITION ENGINE
# ==========================================
def get_hubble_vac(z):
    """Continuous expansion history exactly as defined in Section 8.3.1"""
    # 1. Evaluate S(z) for the transition
    # 1 locally (z ~ 0), 0 in the early universe (z > 0.641)
    arg = (Z_TRANS - z) / WIDTH
    S_z = np.where(arg > 100, 1.0, np.where(arg < -100, 0.0, 1.0 / (1.0 + np.exp(-arg))))
    
    # 2. Interpolate Constitutive Parameters
    OM_Z = OM_EARLY + (OM_LATE - OM_EARLY) * S_z
    OL_Z = 1.0 - OM_Z
    H_Z  = H_EARLY + (H_LATE - H_EARLY) * S_z
    
    # 3. Calculate dynamic Hubble parameter
    E_z = np.sqrt(OM_Z * (1 + z)**3 + OL_Z)
    return H_Z * E_z

# High-resolution Grid
z_grid = np.linspace(0, 2.0, 5000)

# ==========================================
# 3. CALCULATE DERIVATIVES (q and j)
# ==========================================
# VED Kinematics
Hz = get_hubble_vac(z_grid)
dHdz = np.gradient(Hz, z_grid)
q = (1 + z_grid) * (dHdz / Hz) - 1
dqdz = np.gradient(q, z_grid)
j_vac = q * (2 * q + 1) + (1 + z_grid) * dqdz

# Standard LCDM Baseline (For Comparison)
Hz_std = 67.36 * np.sqrt(0.3153 * (1 + z_grid)**3 + 0.6847)
dHdz_std = np.gradient(Hz_std, z_grid)
q_std = (1 + z_grid) * (dHdz_std / Hz_std) - 1
dqdz_std = np.gradient(q_std, z_grid)
j_std = q_std * (2 * q_std + 1) + (1 + z_grid) * dqdz_std

# ==========================================
# 4. ANALYSIS & VERDICT
# ==========================================
# Slice off the first and last 10 points to remove np.gradient boundary artifacts
max_j_vac = np.max(np.abs(j_vac[10:-10]))
max_j_std = np.max(np.abs(j_std[10:-10]))

print(f"\nMax Absolute Jerk (Standard LCDM): {max_j_std:.2f}")
print(f"Max Absolute Jerk (VED Transition): {max_j_vac:.2f}")
print("-" * 50)

if max_j_vac < 2.0:
    print("[ PASS ] The Phase Transition is Adiabatically Smooth.")
    print(f"The Viscoelastic Relaxation Time (Transition Width dz={WIDTH})")
    print("successfully cushions the macroscopic jamming transition, preventing")
    print("any kinematic singularities (Cosmic Whiplash).")
else:
    print("[ FAIL ] Whiplash detected.")

# ==========================================
# 5. PLOT
# ==========================================
plt.figure(figsize=(9, 6))
plt.plot(z_grid, j_vac, 'r-', linewidth=3, label='Vacuum Elastodynamics $j(z)$')
plt.plot(z_grid, j_std, 'k--', linewidth=2, label=r'Standard $\Lambda$CDM Baseline')

plt.axvline(Z_TRANS, color='gray', linestyle=':', label=rf'Phase Transition ($z={Z_TRANS}$)')
plt.xlabel('Redshift $z$', fontsize=12)
plt.ylabel('Jerk Parameter $j(z)$', fontsize=12)
plt.title('Kinematic Stability: Adiabatic Smoothness of the Vacuum Transition', fontsize=14)
plt.ylim(0, 1.5)
plt.legend(loc='lower right', fontsize=11)
plt.grid(True, alpha=0.3)
plt.gca().invert_xaxis() # Standard cosmological convention (past to present)

plt.tight_layout()
plt.savefig('Jerk_Stability_Test.png', dpi=300)
print("\nPlot saved as 'Jerk_Stability_Test.png'")
plt.show()
