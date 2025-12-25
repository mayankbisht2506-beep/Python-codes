import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# ==========================================
# 1. DATA: f*sigma8 Measurements (Gold Dataset)
# ==========================================
fs8_data = np.array([
    [0.067, 0.423, 0.055], [0.17, 0.51, 0.06], [0.22, 0.42, 0.07],
    [0.25, 0.351, 0.058], [0.37, 0.460, 0.038], [0.41, 0.44, 0.07],
    [0.57, 0.450, 0.035], [0.60, 0.43, 0.04], [0.78, 0.38, 0.04],
    [0.80, 0.47, 0.08]
])
z_data = fs8_data[:, 0]
fs8_obs = fs8_data[:, 1]
fs8_err = fs8_data[:, 2]

# ==========================================
# 2. PHYSICS MODEL (Vacuum Phase Transition)
# ==========================================
Om0 = 0.315
sigma8_0_planck = 0.811

# The Fix: Superfluid -> Glass Transition
# Early Universe (Melt): Viscosity ~ 0 (Frictionless flow)
# Late Universe (Glass): Viscosity ~ 0.17 (Lattice Friction)
ETA_EARLY = 0.00
ETA_LATE = 0.17
Z_TRANS = 0.65
WIDTH = 0.15

def sigmoid_safe(x):
    x = np.clip(x, -100, 100)
    return 1.0 / (1.0 + np.exp(-x))

def get_viscosity(z):
    """
    Returns Viscosity eta(z).
    z > 0.65: Superfluid (eta ~ 0)
    z < 0.65: Glassy (eta ~ 0.17)
    """
    # Transition Logic
    arg = (Z_TRANS - z) / WIDTH
    step = sigmoid_safe(arg)

    # step is 1 at Low z (Late), 0 at High z (Early)
    return ETA_EARLY * (1.0 - step) + ETA_LATE * step

def hubble_norm(a):
    z = 1.0/a - 1.0
    return np.sqrt(Om0 * (1+z)**3 + (1-Om0))

def growth_equation_consistent(y, a, model='lcdm'):
    delta, delta_prime = y
    z = 1.0/a - 1.0
    E = hubble_norm(a)

    # Base Friction (LCDM)
    dEda = -1.5 * Om0 * (a**-4) / E
    friction = 3.0/a + dEda/E

    # Viscous Term (Only active in 'full' model)
    if model == 'full':
        friction += get_viscosity(z) / a

    # Source Term (Standard Geometric Consistency)
    # We assume G-boost and H-boost cancel in the background metric
    source = 1.5 * (Om0 / a**3) / (E**2) / a**2

    d2_delta = - friction * delta_prime + source * delta
    return [delta_prime, d2_delta]

# ==========================================
# 3. RUN SIMULATION
# ==========================================
a_eval = np.linspace(0.001, 1.0, 500)
y0 = [1e-3, 1.0]

# Standard LCDM
sol_lcdm = odeint(growth_equation_consistent, y0, a_eval, args=('lcdm',))
delta_lcdm = sol_lcdm[:, 0]
f_lcdm = a_eval * sol_lcdm[:, 1] / delta_lcdm
fs8_lcdm = f_lcdm * sigma8_0_planck * (delta_lcdm / delta_lcdm[-1])

# Vacuum Elastodynamics
sol_full = odeint(growth_equation_consistent, y0, a_eval, args=('full',))
delta_full = sol_full[:, 0]
f_full = a_eval * sol_full[:, 1] / delta_full
fs8_full = f_full * sigma8_0_planck * (delta_full / delta_lcdm[-1])

# ==========================================
# 4. RESULTS
# ==========================================
fs8_model_lcdm = np.interp(1/(1+z_data), a_eval, fs8_lcdm)
fs8_model_full = np.interp(1/(1+z_data), a_eval, fs8_full)

chi2_lcdm = np.sum(((fs8_obs - fs8_model_lcdm)/fs8_err)**2)
chi2_full = np.sum(((fs8_obs - fs8_model_full)/fs8_err)**2)
d_chi2 = chi2_full - chi2_lcdm

suppression = delta_full[-1] / delta_lcdm[-1]
s8_pred = sigma8_0_planck * suppression

print(f"--- SUPERFLUID VACUUM RESULTS ---")
print(f"Planck LCDM Chi2:   {chi2_lcdm:.2f}")
print(f"Vacuum Model Chi2:  {chi2_full:.2f}")
print(f"Delta Chi2:         {d_chi2:.2f} (Target: -1.81)")
print(f"Predicted S8:       {s8_pred:.3f} (Target: ~0.71)")

# Plot
plt.figure(figsize=(10, 6))
plt.errorbar(z_data, fs8_obs, yerr=fs8_err, fmt='o', color='black', label='Data (BOSS/WiggleZ)')
plt.plot(1/a_eval - 1, fs8_lcdm, 'k--', label=f'Planck LCDM ($S_8=0.83$)')
plt.plot(1/a_eval - 1, fs8_full, 'r-', linewidth=2.5, label=f'Vacuum Elastodynamics ($S_8={s8_pred:.2f}$)')
plt.xlim(0, 1.4)
plt.ylim(0.2, 0.6)
plt.xlabel('Redshift z')
plt.ylabel(r'$f\sigma_8(z)$')
plt.title(f'Resolution of S8 Tension ($\Delta\chi^2={d_chi2:.2f}$)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.gca().invert_xaxis()
plt.savefig('Figure3_S8_Final.png')
plt.show()
