import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

# ==========================================
# 1. OBSERVATIONAL DATA (BOSS DR12)
# ==========================================
# Alam et al. (2017) - "Tension Subset" as referenced in Add (46)
boss_z = [0.38, 0.51, 0.61]
boss_DM_rd = [10.23, 13.36, 15.45]
boss_DM_err = [0.17, 0.21, 0.22]

# ==========================================
# 2. PHYSICS SETUP (Corrected to Add 46)
# ==========================================
c_light = 299792.458

# --- A. Standard Model (Planck 2018) ---
H0_std = 67.4
rs_std = 147.09 

# --- B. Vacuum Elastodynamics (Add 46) ---
# 1. Local Hubble Constant (Predicted)
# "Predicts a local Hubble constant of H0 approx 74.5" [cite: 10]
H0_vac_local = 74.5 
H0_vac_early = 67.4 # Relaxes to Planck baseline at high z [cite: 753]

# 2. Viscous Horizon Contraction
# "This analytical approximation yields a contraction factor of 0.924" [cite: 974]
# rs_vac is NOT just 1/H0 scaled; it is physically damped by viscosity.
contraction_factor = 0.924
rs_vac = rs_std * contraction_factor

# 3. Stiffness Phase Transition Parameters
# "Transition epoch... derived... as the geometric Percolation Threshold z approx 0.65" [cite: 41, 154]
z_trans = 0.65
delta_z = 0.1   # Width of transition (approximate from Fig 2)

print(f"--- THEORY CHECK (Add 46) ---")
print(f"Local H0:      {H0_vac_local} km/s/Mpc")
print(f"Transition z:  {z_trans} (Percolation Threshold)")
print(f"rs Contracted: {rs_vac:.2f} Mpc (Factor {contraction_factor})")

# Standard Expansion History E(z)
def E_std(z):
    # Standard Omega_m = 0.315 [cite: 147]
    return np.sqrt(0.315*(1+z)**3 + 0.685)

# Dynamic Hubble Parameter H(z) for Vacuum Model
# Implements the "Stiff Vacuum Transition" (Fig 2) [cite: 741]
def get_H_vac(z):
    # Sigmoidal relaxation between Local (74.5) and Early (67.4)
    # Eq (12) describes stiffness relaxation; H(z) follows the inverse trend.
    # At z << 0.65, H -> 74.5. At z >> 0.65, H -> 67.4.
    
    # Weight w goes from 1.0 (at z=0) to 0.0 (at high z)
    w = 1.0 / (1.0 + np.exp((z - z_trans) / delta_z))
    
    # Interpolate H0
    H0_dynamic = H0_vac_local * w + H0_vac_early * (1 - w)
    
    return H0_dynamic * E_std(z)

def integrand_vac(z):
    return c_light / get_H_vac(z)

def integrand_std(z):
    return c_light / (H0_std * E_std(z))

def get_distance_ratios(model='std'):
    z_grid = np.linspace(0.1, 0.8, 100)
    DM_rd_list = []
    
    # Select sound horizon ruler
    rs = rs_std if model == 'std' else rs_vac
    
    for z in z_grid:
        if model == 'std':
            integral, _ = quad(integrand_std, 0, z)
        else:
            # Vacuum Model integrates dynamic H(z)
            integral, _ = quad(integrand_vac, 0, z)
            
        DM = integral
        DM_rd_list.append(DM / rs)
        
    return z_grid, np.array(DM_rd_list)

# ==========================================
# 3. RUN TEST
# ==========================================
z_model, ratio_std = get_distance_ratios('std')
_, ratio_vac = get_distance_ratios('vac')

# ==========================================
# 4. PLOT
# ==========================================
plt.figure(figsize=(10, 7))

# Plot Data
plt.errorbar(boss_z, boss_DM_rd, yerr=boss_DM_err, fmt='o', color='black', 
             label='BOSS DR12 Data (Tension Subset)', capsize=5, zorder=3)

# Plot Models
plt.plot(z_model, ratio_std, 'b--', linewidth=2, label=f'Standard LCDM (H0={H0_std})')
plt.plot(z_model, ratio_vac, 'r-', linewidth=3, alpha=0.8, label=f'Vacuum Elastodynamics (Dynamic H0)')

# Highlight Transition Zone
plt.axvspan(z_trans - 0.05, z_trans + 0.05, color='red', alpha=0.1, label='Percolation Transition (z=0.65)')

plt.xlabel('Redshift $z$', fontsize=12)
plt.ylabel(r'Transverse BAO Distance $D_M(z) / r_d$', fontsize=12)
plt.title(f'BAO Consistency Check: Vacuum Phase Transition\n(Add 46 Implementation)', fontsize=14)
plt.legend(fontsize=11, loc='upper left')
plt.grid(True, alpha=0.3)

# Annotate Logic
plt.annotate(f"Mechanism:\n1. H0 relaxes {H0_vac_local} -> {H0_vac_early}\n2. rs contracts by {100*(1-contraction_factor):.1f}%", 
             xy=(0.45, 11), xycoords='data',
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="firebrick", alpha=0.9),
             color='firebrick', fontsize=10)

plt.tight_layout()
plt.savefig('Figure_BAO_Add46_Corrected.png')
plt.show()

# Verification Calculation
print("\n--- CHECKPOINTS (Add 46) ---")
for idx, z in enumerate(boss_z):
    target = boss_DM_rd[idx]
    val_std = np.interp(z, z_model, ratio_std)
    val_vac = np.interp(z, z_model, ratio_vac)
    print(f"z={z}: Data={target:.2f} | Std={val_std:.2f} | Vac={val_vac:.2f} (Diff: {val_vac-val_std:.3f})")
