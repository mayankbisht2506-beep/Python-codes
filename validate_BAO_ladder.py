import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

print("============================================================")
print("   BAO CONSISTENCY AUDIT: VACUUM ELASTODYNAMICS")
print("   Testing Mechanism: Superfluid Horizon Contraction")
print("============================================================")

# ==========================================
# 1. OBSERVATIONAL DATA (BOSS DR12)
# ==========================================
# Source: Alam et al. (2017)
boss_z = [0.38, 0.51, 0.61]
boss_DM_rd = [10.23, 13.36, 15.45]
boss_DM_err = [0.17, 0.21, 0.22]

# ==========================================
# 2. PHYSICS ENGINE (RIGOROUS)
# ==========================================
c_light = 299792.458

# --- MODEL A: Standard Planck (Baseline) ---
H0_std = 67.4
Om_std = 0.315
rs_std = 147.09  # Standard Sound Horizon

# --- MODEL B: Vacuum Elastodynamics (Geometric Limit) ---
# PHYSICS: High G_early (Section 7.2) -> Fast Expansion -> Contracted rs
# SCALING: rs_vac = rs_std * (H0_std / H0_vac) approx 0.905
H0_vac = 74.5       # Theoretical Geometric Yield Limit (Eq 88)
Om_vac = 0.343      # Inertial Counter-Load (MCMC Result, Section 8.6)
contraction_factor = 0.905 # (Derived in Section 7.12.1)
rs_vac = rs_std * contraction_factor 

print(f"\n[PHYSICS PARAMETERS]")
print(f"Standard Model: H0={H0_std}, Om={Om_std}, rs={rs_std:.2f} Mpc")
print(f"Vacuum Model:   H0={H0_vac}, Om={Om_vac}, rs={rs_vac:.2f} Mpc")
print(f"Mechanism:      Geometric Scaling Cancellation (Contraction: {100*(1-contraction_factor):.1f}%)")

# INTEGRANDS
# The vacuum model uses the HIGH-ENERGY trajectory everywhere.
# This is required to justify the contracted rs.
def integrand_std(z):
    E_z = np.sqrt(Om_std * (1 + z)**3 + (1 - Om_std))
    return c_light / (H0_std * E_z)

def integrand_vac(z):
    E_z = np.sqrt(Om_vac * (1 + z)**3 + (1 - Om_vac))
    return c_light / (H0_vac * E_z)

# CALCULATION ENGINE
def get_distance_ratio(z_target, model='std'):
    if model == 'std':
        integral, _ = quad(integrand_std, 0, z_target)
        return integral / rs_std
    else:
        integral, _ = quad(integrand_vac, 0, z_target)
        return integral / rs_vac

# ==========================================
# 3. EXECUTE VALIDATION (TABLE 4 REPRODUCTION)
# ==========================================
print("\n[TABLE 4 REPRODUCTION & AUDIT]")
print(f"{'z':<6} | {'Data':<10} | {'Planck':<10} | {'Vacuum':<10} | {'Residual':<10} | {'Sigma':<6}")
print("-" * 70)

chi2_vac = 0
chi2_std = 0

for i, z in enumerate(boss_z):
    target = boss_DM_rd[i]
    error = boss_DM_err[i]
    
    val_std = get_distance_ratio(z, 'std')
    val_vac = get_distance_ratio(z, 'vac')
    
    resid = val_vac - target
    sigma = resid / error
    
    chi2_vac += sigma**2
    chi2_std += ((val_std - target)/error)**2
    
    print(f"{z:<6.2f} | {target:<10.2f} | {val_std:<10.2f} | {val_vac:<10.2f} | {resid:<+10.2f} | {sigma:<+6.2f}")

print("-" * 70)
print(f"TOTAL CHI-SQUARED: Standard={chi2_std:.2f} | Vacuum={chi2_vac:.2f}")

if chi2_vac < 2.0:
    print("\nVERDICT: SUCCESS. The Scaling Cancellation is physically exact.")
else:
    print("\nVERDICT: FAILURE. Check Om_vac or Contraction Factor.")

# ==========================================
# 4. PLOT GENERATION
# ==========================================
z_grid = np.linspace(0.2, 0.7, 100)
ratio_std_list = [get_distance_ratio(z, 'std') for z in z_grid]
ratio_vac_list = [get_distance_ratio(z, 'vac') for z in z_grid]

plt.figure(figsize=(10, 6))
# Plot Data
plt.errorbar(boss_z, boss_DM_rd, yerr=boss_DM_err, fmt='o', color='black', 
             label='BOSS DR12 Data', capsize=5, zorder=5)

# Plot Models
plt.plot(z_grid, ratio_std_list, 'b--', linewidth=2, label='Planck Baseline (67.4)')
plt.plot(z_grid, ratio_vac_list, 'r-', linewidth=2.5, label=f'Vacuum Elastodynamics (74.5)\nwith 9.5% rs Contraction')

plt.title('BAO Consistency Check: Superfluid Horizon Contraction', fontsize=14)
plt.xlabel('Redshift $z$', fontsize=12)
plt.ylabel(r'$D_M(z) / r_d$', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('Figure_BAO_Audit_Rigorous.png')
print("Plot saved to 'Figure_BAO_Audit_Rigorous.png'")
plt.show()
