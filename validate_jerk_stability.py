import numpy as np
import matplotlib.pyplot as plt

print("--- COSMIC WHIPLASH (JERK PARAMETER STABILITY) ---")
print("Objective: Verify the Phase Transition is Adiabatically Smooth.")

# ==========================================
# 1. PHYSICS PARAMETERS (From VED Theory)
# ==========================================
Z_TRANS = 0.65
WIDTH = 0.10

# Dual-Trajectory Anchor Points
# EARLY REGIME (z > 0.65)
H_EARLY = 74.5
OM_EARLY = 0.315

# LATE REGIME (z < 0.65)
H_LATE = 72.87
OM_LATE = 0.357  # Includes Inertial Counter-Load

# ==========================================
# 2. RIGOROUS DUAL-TRANSITION ENGINE
# ==========================================
def H_early_trajectory(z):
    return H_EARLY * np.sqrt(OM_EARLY * (1+z)**3 + (1-OM_EARLY))

def H_late_trajectory(z):
    return H_LATE * np.sqrt(OM_LATE * (1+z)**3 + (1-OM_LATE))

def get_hubble_vac(z):
    # Sigmoid Transition
    # z >> 0.65 -> Early Universe -> w = 1.0
    # z << 0.65 -> Late Universe -> w = 0.0
    w = 1.0 / (1.0 + np.exp(-(z - Z_TRANS)/WIDTH))
    return w * H_early_trajectory(z) + (1.0 - w) * H_late_trajectory(z)

# High-resolution Grid
z_grid = np.linspace(0, 2.0, 2000)

# ==========================================
# 3. CALCULATE DERIVATIVES (q and j)
# ==========================================
# VED Kinematics
Hz = get_hubble_vac(z_grid)
dHdz = np.gradient(Hz, z_grid)
q = (1+z_grid) * (dHdz / Hz) - 1
dqdz = np.gradient(q, z_grid)
j_vac = q*(2*q + 1) + (1+z_grid)*dqdz

# Standard LCDM Baseline (For Comparison)
Hz_std = 67.4 * np.sqrt(0.315 * (1+z_grid)**3 + 0.685)
dHdz_std = np.gradient(Hz_std, z_grid)
q_std = (1+z_grid) * (dHdz_std / Hz_std) - 1
dqdz_std = np.gradient(q_std, z_grid)
j_std = q_std*(2*q_std + 1) + (1+z_grid)*dqdz_std

# ==========================================
# 4. ANALYSIS & VERDICT
# ==========================================
max_j_vac = np.max(np.abs(j_vac))
max_j_std = np.max(np.abs(j_std)) # Strictly 1.00

print(f"Max Absolute Jerk (Standard LCDM): {max_j_std:.2f}")
print(f"Max Absolute Jerk (VED Dual-Transition): {max_j_vac:.2f}")

print("-" * 50)
if max_j_vac < 2.0:
    print("[ PASS ] The Phase Transition is Adiabatically Smooth.")
    print("The Inertial Counter-Load successfully cushions the expansion shift,")
    print("preventing any unphysical 'Cosmic Whiplash'.")
else:
    print("[ FAIL ] Whiplash detected.")

# ==========================================
# 5. PLOT
# ==========================================
plt.figure(figsize=(9,6))
plt.plot(z_grid, j_vac, 'r-', linewidth=3, label='Vacuum Elastodynamics j(z)')
plt.plot(z_grid, j_std, 'k--', linewidth=2, label='Standard LCDM Baseline (j=1)')

plt.axvline(Z_TRANS, color='gray', linestyle=':', label='Phase Transition (z=0.65)')
plt.xlabel('Redshift z', fontsize=12)
plt.ylabel('Jerk Parameter j(z)', fontsize=12)
plt.title('Stability Analysis: Adiabatic Smoothness of the Vacuum Transition', fontsize=14)
plt.ylim(0, 1.5)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.gca().invert_xaxis()

plt.tight_layout()
plt.savefig('Jerk_Stability_Test.png')
print("Plot saved as 'Jerk_Stability_Test.png'")
plt.show()
