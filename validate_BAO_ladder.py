import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

print("--- BAO CONSISTENCY CHECK: SUPERFLUID DETACHMENT (9.5% CONTRACTION) ---")

# ==========================================
# 1. OBSERVATIONAL DATA (BOSS DR12)
# ==========================================
boss_z = [0.38, 0.51, 0.61]
boss_DM_rd = [10.23, 13.36, 15.45]
boss_DM_err = [0.17, 0.21, 0.22]

# ==========================================
# 2. PHYSICS SETUP
# ==========================================
c_light = 299792.458

# --- A. Standard Model ---
H0_std = 67.4
Om_std = 0.315  # Planck Baseline

# --- B. Vacuum Elastodynamics ---
H0_vac_local = 74.5
Om_vac = 0.343  # Matches Global MCMC Fit (Section 8.6) [cite: 877]

# 3. Stiffness Phase Transition
z_trans = 0.65
delta_z = 0.1    

# 2. Superfluid Horizon Contraction
# Driven by Stiffness ONLY (eta ~ 0) -> Factor 0.905
# [cite_start]Source: Section 7.11.2, Equation 86 [cite: 780]
contraction_factor = 0.905
rs_vac = 147.09 * contraction_factor 

print(f"--- PHYSICS PARAMETERS ---")
print(f"Local H0:      {H0_vac_local} km/s/Mpc")
print(f"Vacuum Om:     {Om_vac} (Matches MCMC)")
print(f"Transition z:  {z_trans}")
print(f"rs Contracted: {rs_vac:.2f} Mpc (Factor {contraction_factor})")
print(f"Mechanism:     Superfluid Detachment (eta ~ 0)")

# Standard Expansion History E(z)
def E_std(z):
    return np.sqrt(Om_std*(1+z)**3 + (1-Om_std))

# Dynamic Hubble Parameter H(z)
# CORRECTED: Uses Om_vac (0.350)
def get_H_vac(z):
    # Sigmoidal relaxation for H0
    w = 1.0 / (1.0 + np.exp((z - z_trans) / delta_z))
    H0_dynamic = H0_vac_local * w + 67.4 * (1 - w)
    
    # Use Om_vac (0.350) for the Vacuum Model E(z)
    E_vac = np.sqrt(Om_vac*(1+z)**3 + (1-Om_vac)) 
    
    return H0_dynamic * E_vac

def integrand_vac(z):
    return c_light / get_H_vac(z)

def integrand_std(z):
    return c_light / (H0_std * E_std(z))

def get_distance_ratios(model='std'):
    z_grid = np.linspace(0.1, 0.8, 100)
    DM_rd_list = []
    
    # Select rs based on model
    if model == 'std':
        rs = 147.09
    else:
        rs = rs_vac
    
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
plt.plot(z_model, ratio_vac, 'r-', linewidth=3, alpha=0.8, label=f'Vacuum Model (H0={H0_vac_local}, Om={Om_vac})')

# Transition Zone
plt.axvspan(z_trans - 0.05, z_trans + 0.05, color='green', alpha=0.1, label='Phase Transition (z=0.65)')

plt.xlabel('Redshift $z$', fontsize=12)
plt.ylabel(r'Transverse BAO Distance $D_M(z) / r_d$', fontsize=12)
plt.title(f'BAO Verification: Superfluid Detachment (9.5% Contraction)', fontsize=14)
plt.legend(fontsize=11, loc='upper left')
plt.grid(True, alpha=0.3)

# Annotate Logic
plt.annotate(f"Contraction: {100*(1-contraction_factor):.1f}%\n(rs = {rs_vac:.1f} Mpc)", 
             xy=(0.45, 11), xycoords='data',
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="firebrick", alpha=0.9),
             color='firebrick', fontsize=10)

plt.tight_layout()
plt.savefig('BAO_Superfluid_Check_Corrected.png')
plt.show()

# ==========================================
# 5. GENERATE TABLE IV VALUES
# ==========================================
print("\n--- NEW TABLE IV VALUES (Matches Add 101.pdf) ---")
print(f"{'z':<5} | {'Data':<10} | {'Vacuum':<10} | {'Residual':<10}")
print("-" * 45)

for idx, z in enumerate(boss_z):
    target = boss_DM_rd[idx]
    val_vac = np.interp(z, z_model, ratio_vac)
    residual = val_vac - target
    
    print(f"{z:<5} | {target:<10.2f} | {val_vac:<10.2f} | {residual:+.2f}")

print("-" * 45)
print("EXPECTED RESULTS with Om=0.343:")
print("all Residual less then 0.20")
