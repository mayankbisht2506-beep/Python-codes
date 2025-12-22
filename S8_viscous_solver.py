import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# ==========================================
# 1. DATA: f*sigma8 Measurements (Gold Dataset)
# ==========================================
# Format: [z, fsigma8, error]
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
# 2. PHYSICS MODELS
# ==========================================
Om0 = 0.315
sigma8_0_planck = 0.811
ETA_DRAG = 0.17
Z_TRANS = 0.65
WIDTH = 0.15

def get_viscosity(z):
    # Viscosity turns on during lattice relaxation (Low z)
    return ETA_DRAG / (1 + np.exp((z - Z_TRANS) / WIDTH))

def hubble_norm(a):
    z = 1.0/a - 1.0
    return np.sqrt(Om0 * (1+z)**3 + (1-Om0))

def growth_equation_dynamic(y, a, model='lcdm'):
    delta, delta_prime = y
    z = 1.0/a - 1.0
    E = hubble_norm(a)

    # --- BUG FIX: dE/da must be NEGATIVE ---
    # E^2 = Om/a^3 + ... => 2E dE/da = -3 Om/a^4
    dEda = -1.5 * Om0 * (a**-4) / E  # Corrected Sign

    # Base Friction
    friction = 3.0/a + dEda/E

    # Viscous Boost (Only for 'viscous' model)
    if model == 'viscous':
        friction += get_viscosity(z) / a

    # Source Term
    source = 1.5 * (Om0 / a**3) / (E**2) / a**2

    d2_delta = - friction * delta_prime + source * delta
    return [delta_prime, d2_delta]

# Solve ODEs
a_eval = np.linspace(0.001, 1.0, 500)
y0 = [1e-3, 1.0] # Growing mode IC

# 1. LCDM Solution
sol_lcdm = odeint(growth_equation_dynamic, y0, a_eval, args=('lcdm',))
delta_lcdm = sol_lcdm[:, 0]
f_lcdm = a_eval * sol_lcdm[:, 1] / delta_lcdm
fs8_lcdm = f_lcdm * sigma8_0_planck * (delta_lcdm / delta_lcdm[-1])

# 2. Viscous Solution
sol_visc = odeint(growth_equation_dynamic, y0, a_eval, args=('viscous',))
delta_visc = sol_visc[:, 0]
f_visc = a_eval * sol_visc[:, 1] / delta_visc
# Scale relative to Planck baseline to show suppression
fs8_visc = f_visc * sigma8_0_planck * (delta_visc / delta_lcdm[-1])

# --- 3. STATISTICS ---
fs8_model_lcdm = np.interp(1/(1+z_data), a_eval, fs8_lcdm)
fs8_model_visc = np.interp(1/(1+z_data), a_eval, fs8_visc)

chi2_lcdm = np.sum(((fs8_obs - fs8_model_lcdm)/fs8_err)**2)
chi2_visc = np.sum(((fs8_obs - fs8_model_visc)/fs8_err)**2)

# AIC (k=0 for both, fixed parameters)
aic_lcdm = chi2_lcdm
aic_visc = chi2_visc

print(f"--- FINAL S8 RESULTS (BUG FIXED) ---")
print(f"Planck LCDM Chi2: {chi2_lcdm:.2f}")
print(f"Viscous Vacuum Chi2: {chi2_visc:.2f}")
print(f"Delta Chi2: {chi2_visc - chi2_lcdm:.2f}")
if chi2_visc < chi2_lcdm:
    print("SUCCESS: Viscous model is statistically preferred.")
else:
    print("CHECK: Viscous model still has higher Chi2.")

# --- 4. PLOT ---
plt.figure(figsize=(10, 6))
plt.errorbar(z_data, fs8_obs, yerr=fs8_err, fmt='o', color='black', label='Data (BOSS/WiggleZ/6dF)')
plt.plot(1/a_eval - 1, fs8_lcdm, 'k--', label='Planck LCDM (Baseline)')
plt.plot(1/a_eval - 1, fs8_visc, 'r-', linewidth=2.5, label='Viscous Vacuum (Prediction)')
plt.xlim(0, 1.5)
plt.ylim(0.2, 0.6)
plt.xlabel('Redshift z')
plt.ylabel('$f\sigma_8(z)$')
plt.title('Figure 3: Resolution of S8 Tension via Viscosity')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('Figure3_S8_Validation_Fixed.png')
plt.show()
