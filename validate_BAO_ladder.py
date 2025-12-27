import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

# ==========================================
# 1. OBSERVATIONAL DATA (BOSS DR12)
# ==========================================
# Alam et al. (2017) Consensus BAO
# z_eff | D_M/r_d | D_H/r_d (= c/H/r_d)
# We use the D_V/r_d combo for simplicity or the transversal/radial split.
# Let's use the D_V/r_s standard quantity.
# D_V = [z * D_M^2 * D_H]^1/3
# Data: z, D_V/r_d, Error
bao_data = np.array([
    [0.38, 1477/147.78, 0.02], # Approx normalized to fiducial r_d
    [0.51, 1877/147.78, 0.02],
    [0.61, 2140/147.78, 0.02]
])
# Note: Real BOSS data is usually D_M and D_H. 
# Let's use the exact values from Alam 2017 relative to r_d.
# z=0.38: D_M/r_d = 10.23 +/- 0.17, D_H/r_d = 25.00 +/- 0.76
# z=0.51: D_M/r_d = 13.36 +/- 0.21, D_H/r_d = 22.33 +/- 0.58
# z=0.61: D_M/r_d = 15.45 +/- 0.22, D_H/r_d = 20.06 +/- 0.45

boss_z = [0.38, 0.51, 0.61]
boss_DM_rd = [10.23, 13.36, 15.45]
boss_DM_err = [0.17, 0.21, 0.22]

# ==========================================
# 2. PHYSICS SETUP
# ==========================================
c_light = 299792.458 # km/s

# 1. Standard Planck
H0_std = 67.4
rs_std = 147.09 # Mpc (Planck 2018)

# 2. Vacuum Elastodynamics
H0_vac = 73.0
# From your Section 7.8 result:
# Contraction factor 0.917 -> rs_vac = 147.09 * 0.917
rs_vac = 147.09 * 0.917 

def E_inv_std(z):
    return 1.0/np.sqrt(0.315*(1+z)**3 + 0.685)

# Vacuum Expansion History (Viscous Drag)
# Matches SNe best fit (approx Om=0.315, but different H0)
# Note: Distances scale as 1/H0.
def get_distance_ratios(model='std'):
    if model == 'std':
        H0 = H0_std
        rs = rs_std
    else:
        H0 = H0_vac
        rs = rs_vac
        
    z_grid = np.linspace(0.1, 0.8, 100)
    DM_rd_list = []
    
    for z in z_grid:
        # Comoving Distance D_M
        # Integral dz/E(z)
        integral, _ = quad(E_inv_std, 0, z)
        DM = (c_light / H0) * integral
        
        # Ratio D_M / r_d
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
plt.figure(figsize=(8, 6))

# Plot Data
plt.errorbar(boss_z, boss_DM_rd, yerr=boss_DM_err, fmt='o', color='black', 
             label='BOSS DR12 Data (Galaxies)', capsize=5, zorder=3)

# Plot Models
plt.plot(z_model, ratio_std, 'b--', linewidth=2, label=f'Standard LCDM (H0={H0_std})')
plt.plot(z_model, ratio_vac, 'r-', linewidth=3, label=f'Vacuum Elastodynamics (H0={H0_vac})')

plt.xlabel('Redshift $z$', fontsize=12)
plt.ylabel(r'Transverse BAO Distance $D_M(z) / r_d$', fontsize=12)
plt.title(f'BAO Distance Ladder Test\n(Sound Horizon Contraction Check)', fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)

# Annotate
plt.annotate(f"r_s = {rs_vac:.1f} Mpc", xy=(0.2, 8), color='red', fontsize=12, fontweight='bold')
plt.annotate(f"r_s = {rs_std:.1f} Mpc", xy=(0.2, 9), color='blue', fontsize=12)

plt.tight_layout()
plt.savefig('Figure_BAO_Ladder.png')
print("Plot saved as 'Figure_BAO_Ladder.png'")
plt.show()

# Verification Calculation at z=0.51
def check_point(idx):
    z = boss_z[idx]
    target = boss_DM_rd[idx]
    
    # STD
    val_std = np.interp(z, z_model, ratio_std)
    
    # VAC
    val_vac = np.interp(z, z_model, ratio_vac)
    
    print(f"z={z}: Data={target:.2f} | Std={val_std:.2f} | Vac={val_vac:.2f}")

print("\n--- BAO CHECKPOINTS ---")
check_point(0)
check_point(1)
check_point(2)
