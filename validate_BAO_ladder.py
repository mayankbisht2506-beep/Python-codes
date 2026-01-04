import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

print("--- BAO CONSISTENCY CHECK: FINAL UPDATED MODEL (8.1% CONTRACTION) ---")

# ==========================================
# 1. OBSERVATIONAL DATA (BOSS DR12)
# ==========================================
boss_z = [0.38, 0.51, 0.61]
boss_DM_rd = [10.23, 13.36, 15.45]
boss_DM_err = [0.17, 0.21, 0.22]

# ==========================================
# 2. PHYSICS SETUP (Matches Updated Section 7.11)
# ==========================================
c_light = 299792.458

# --- A. Standard Model (Planck 2018) ---
H0_std = 67.4
rs_std = 147.09 

# --- B. Vacuum Elastodynamics ---
# 1. Local Hubble Constant (Matches SH0ES)
H0_vac_local = 74.5 
H0_vac_early = 67.4 

# 2. Viscous Horizon Contraction (UPDATED)
# Driven by Lepton Load (eta ~ 0.157) -> Factor 0.919
contraction_factor = 0.919
rs_vac = rs_std * contraction_factor

# 3. Stiffness Phase Transition
z_trans = 0.65
delta_z = 0.1   

print(f"--- PHYSICS PARAMETERS ---")
print(f"Local H0:      {H0_vac_local} km/s/Mpc")
print(f"Transition z:  {z_trans}")
print(f"rs Contracted: {rs_vac:.2f} Mpc (Factor {contraction_factor})")
print(f"Mechanism:     Lepton Load Viscosity (eta ~ 0.157)")

# Standard Expansion History E(z)
def E_std(z):
    return np.sqrt(0.315*(1+z)**3 + 0.685)

# Dynamic Hubble Parameter H(z)
def get_H_vac(z):
    # Sigmoidal relaxation
    w = 1.0 / (1.0 + np.exp((z - z_trans) / delta_z))
    H0_dynamic = H0_vac_local * w + H0_vac_early * (1 - w)
    return H0_dynamic * E_std(z)

def integrand_vac(z):
    return c_light / get_H_vac(z)

def integrand_std(z):
    return c_light / (H0_std * E_std(z))

def get_distance_ratios(model='std'):
    z_grid = np.linspace(0.1, 0.8, 100)
    DM_rd_list = []
    
    rs = rs_std if model == 'std' else rs_vac
    
    for z in z_grid:
        if model == 'std':
            integral, _ = quad(integrand_std, 0, z)
        else:
            integral, _ = quad(integrand_vac, 0, z)
            
        DM = integral
        DM_rd_list.append(DM / rs)
        
    return z_grid, np.array(DM_rd_list)

# ==========================================
# 3. RUN SIMULATION
# ==========================================
z_model, ratio_std = get_distance_ratios('std')
_, ratio_vac = get_distance_ratios('vac')

# ==========================================
# 4. PLOT RESULTS
# ==========================================
plt.figure(figsize=(10, 7))

# Plot Data
plt.errorbar(boss_z, boss_DM_rd, yerr=boss_DM_err, fmt='o', color='black', 
             label='BOSS DR12 Data', capsize=5, zorder=3)

# Plot Models
plt.plot(z_model, ratio_std, 'b--', linewidth=2, label=f'Standard LCDM (H0={H0_std})')
plt.plot(z_model, ratio_vac, 'r-', linewidth=3, alpha=0.8, label=f'Vacuum Model (H0={H0_vac_local})')

# Transition Zone
plt.axvspan(z_trans - 0.05, z_trans + 0.05, color='green', alpha=0.1, label='Stiffness Transition (z=0.65)')

plt.xlabel('Redshift $z$', fontsize=12)
plt.ylabel(r'Transverse BAO Distance $D_M(z) / r_d$', fontsize=12)
plt.title(f'BAO Verification: Lepton Load Model (8.1% Contraction)', fontsize=14)
plt.legend(fontsize=11, loc='upper left')
plt.grid(True, alpha=0.3)

# Annotate Logic
plt.annotate(f"Contraction: {100*(1-contraction_factor):.1f}%\n(rs = {rs_vac:.1f} Mpc)", 
             xy=(0.45, 11), xycoords='data',
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="firebrick", alpha=0.9),
             color='firebrick', fontsize=10)

plt.tight_layout()
plt.show()

# ==========================================
# 5. GENERATE TABLE IV VALUES
# ==========================================
print("\n--- NEW TABLE IV VALUES (To Match Paper) ---")
print(f"{'z':<5} | {'Data':<10} | {'Vacuum':<10} | {'Residual':<10}")
print("-" * 45)

for idx, z in enumerate(boss_z):
    target = boss_DM_rd[idx]
    val_vac = np.interp(z, z_model, ratio_vac)
    residual = val_vac - target
    
    print(f"{z:<5} | {target:<10.2f} | {val_vac:<10.2f} | {residual:+.2f}")

print("-" * 45)
print("VERDICT: These residuals should match your updated Table IV.")
print("(Expected: +0.04, -0.03, +0.11)")
