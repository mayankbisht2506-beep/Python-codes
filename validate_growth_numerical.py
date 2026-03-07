# Uncomment the line below if running in Google Colab / Jupyter
# !pip install scipy numpy matplotlib

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.special import expit  # Safe sigmoid

print("--- GROWTH RATE EVOLUTION: THE RHEOLOGICAL SEPARATION ---")
print("Executing Section 7.6.1: Viscous Background vs. Frictionless Bound States")
print("-" * 75)

# ==========================================
# 1. OBSERVATIONAL DATA (The "Tension Subset")
# ==========================================
data_rsd = np.array([
    [0.38, 0.448, 0.038],  # BOSS Low-z
    [0.51, 0.455, 0.038],  # BOSS Mid-z
    [0.61, 0.410, 0.034],  # BOSS High-z
    [1.48, 0.382, 0.026],  # eBOSS
    [0.44, 0.413, 0.080],  # WiggleZ
    [0.60, 0.390, 0.063],  # WiggleZ
    [0.73, 0.437, 0.072],  # WiggleZ
    [0.60, 0.480, 0.120],  # VIPERS
    [0.86, 0.400, 0.110]   # VIPERS
])

# ==========================================
# 2. PHYSICS PARAMETERS
# ==========================================
H0_LCDM       = 67.36   # EXACT: Planck 2018 Baseline
OM_PLANCK     = 0.3153  # EXACT: Planck 2018 Baseline
SIGMA8_0_LCDM = 0.8111  # EXACT: Planck 2018 Baseline

OM_PRIMORDIAL = 0.3116  # EXACT: Frictionless Bare Density (Drives Gravity)
OM_EFFECTIVE  = 0.3639  # EXACT: Viscous Load (Drives Expansion Friction)

ETA_FLOOR = 0.1569      # EXACT: Lepton Saturation Viscosity
ETA_PEAK  = 0.3116      # EXACT: Simple cubic percolation limit
Z_TRANS   = 0.641       # EXACT: Percolation redshift
WIDTH     = 0.10

def get_viscosity(z):
    arg = (z - Z_TRANS) / WIDTH
    late_trigger = 1.0 - expit(arg)
    base_visc = ETA_FLOOR * late_trigger
    spike = (ETA_PEAK - ETA_FLOOR) * np.exp(-0.5 * ((z - Z_TRANS)/0.15)**2)
    return base_visc + spike

def growth_ode_rigorous(y, a, model='lcdm', env='background'):
    """
    Simulates Section 7.6.1:
    env='background': Full macroscopic friction (Used for Weak Lensing S8)
    env='local_halo': Friction deactivated (eta=0) for bound BOSS galaxies
    """
    delta, delta_prime = y
    z = 1.0/a - 1.0

    if model == 'viscous':
        om_expansion = OM_PRIMORDIAL + (OM_EFFECTIVE - OM_PRIMORDIAL) * expit((Z_TRANS - z) / WIDTH)
        om_gravity = OM_PRIMORDIAL
    else:
        om_expansion = OM_PLANCK
        om_gravity = OM_PLANCK

    E = np.sqrt(om_expansion*(1+z)**3 + (1-om_expansion))
    dE_da = -1.5 * om_expansion * (a**-4) / E

    # Mass-Gravity Cancellation (Sec 7.6.2) perfectly preserves the standard gravity source
    source_term = 1.5 * om_gravity / (a**5 * E**2)

    if model == 'viscous':
        # Apply Rheological Separation Principle
        if env == 'background':
            eta = get_viscosity(z) # Full viscosity for global metric
        else:
            eta = 0.0 # Friction vanishes in bound local halos (theta <= 0)
            
        friction_term = (1.0/a) + (dE_da/E) + (2.0/a) * (1.0 + eta)**2.0
        return [delta_prime, -friction_term * delta_prime + source_term * delta]

    hubble_friction = 3.0/a + dE_da/E
    return [delta_prime, -hubble_friction * delta_prime + source_term * delta]

# ==========================================
# 3. RUN SIMULATIONS
# ==========================================
z_start = 100.0
a_grid = np.linspace(1.0/(1+z_start), 1.0, 500)
y0 = [a_grid[0], 1.0]

sol_lcdm = odeint(growth_ode_rigorous, y0, a_grid, args=('lcdm', 'background'))
sol_vac_bg = odeint(growth_ode_rigorous, y0, a_grid, args=('viscous', 'background'))
sol_vac_local = odeint(growth_ode_rigorous, y0, a_grid, args=('viscous', 'local_halo'))

delta_lcdm = sol_lcdm[:, 0]; d_delta_lcdm = sol_lcdm[:, 1]
delta_vac_bg = sol_vac_bg[:, 0]; d_delta_vac_bg = sol_vac_bg[:, 1]
delta_vac_loc = sol_vac_local[:, 0]; d_delta_vac_loc = sol_vac_local[:, 1]

# Calculate fs8 for LCDM
sig8_lcdm = SIGMA8_0_LCDM * (delta_lcdm / delta_lcdm[-1])
f_lcdm = (a_grid / delta_lcdm) * d_delta_lcdm
fs8_lcdm = f_lcdm * sig8_lcdm

# Calculate S8 for Vacuum Background (Tested against Weak Lensing)
early_scalar = np.sqrt(OM_PRIMORDIAL / OM_PLANCK) 
sig8_vac_bg = (sig8_lcdm[0] * early_scalar / delta_vac_bg[0]) * delta_vac_bg
s8_true_wl = sig8_vac_bg[-1] * np.sqrt(OM_PRIMORDIAL / 0.3)

# Calculate fs8 for Vacuum (Local Halos - Tested against BOSS data)
# THE FIX: We use the frictionless ODE to find the instantaneous rate (f), 
# but we apply it to the true, historically damped global background amplitude (sig8_vac_bg).
f_vac_loc = (a_grid / delta_vac_loc) * d_delta_vac_loc
fs8_vac_loc = f_vac_loc * sig8_vac_bg  # <- The critical correction

# ==========================================
# 4. RESULTS & CHI-SQUARED CHECK
# ==========================================
z_axis = 1.0/a_grid - 1.0

print(f"\n{'Redshift':<10} | {'Data':<15} | {'LCDM':<8} | {'Vacuum (Local)':<15} | {'Status'}")
print("-" * 80)

chi2_lcdm_tot = 0; chi2_vac_tot = 0
data_rsd = data_rsd[data_rsd[:, 0].argsort()]

for row in data_rsd:
    z_val, y_val, err = row
    pred_l = np.interp(z_val, np.flip(z_axis), np.flip(fs8_lcdm))
    pred_v = np.interp(z_val, np.flip(z_axis), np.flip(fs8_vac_loc))

    c2_l = ((pred_l - y_val)/err)**2
    c2_v = ((pred_v - y_val)/err)**2
    chi2_lcdm_tot += c2_l; chi2_vac_tot += c2_v

    status = "BETTER" if abs(pred_v - y_val) < abs(pred_l - y_val) else "WORSE"
    print(f"{z_val:<10.2f} | {y_val:.3f} +/-{err:.3f} | {pred_l:.3f}    | {pred_v:.3f}           | {status}")

print("-" * 80)
print(f"Total Chi2 (LCDM):           {chi2_lcdm_tot:.3f}")
print(f"Total Chi2 (Vacuum Local):   {chi2_vac_tot:.3f}")
print(f"Delta Chi2 (Improvement):    {chi2_vac_tot - chi2_lcdm_tot:.3f}")

print("\n--- WEAK LENSING (BACKGROUND METRIC) ---")
print(f"Viscous Damped Amplitude (sigma_8):  {sig8_vac_bg[-1]:.4f}")
print(f"True Observable Weak Lensing S_8:    {s8_true_wl:.4f}  [DES Y3 Concordance: 0.776]")

# Plot
plt.figure(figsize=(10,6))
plt.plot(z_axis, fs8_lcdm, 'k--', label=rf'Standard $\Lambda$CDM ($\Omega_m={OM_PLANCK}$)')
plt.plot(z_axis, fs8_vac_loc, 'b-', linewidth=2, label=r'Vacuum Local Halos (Frictionless, tested vs BOSS)')

# We calculate the background fs8 just to show it on the graph
f_vac_bg = (a_grid / delta_vac_bg) * d_delta_vac_bg
fs8_vac_bg = f_vac_bg * sig8_vac_bg
plt.plot(z_axis, fs8_vac_bg, 'r-', linewidth=2, alpha=0.5, label=r'Vacuum Background Metric (Viscous, tested vs Lensing)')

plt.errorbar(data_rsd[:,0], data_rsd[:,1], yerr=data_rsd[:,2], fmt='o', color='black', label='RSD Data (Bound Halos)', capsize=3)
plt.xlim(0, 1.6)
plt.xlabel('Redshift z', fontsize=12)
plt.ylabel(r'Growth Rate $f\sigma_8(z)$', fontsize=12)
plt.title('Rheological Separation: Local Galaxies vs Global Background', fontsize=14)
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('Growth_Rheological_Separation.pdf', dpi=300)
print("\nPlot saved as 'Growth_Rheological_Separation.pdf'")
plt.show()
