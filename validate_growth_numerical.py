import numpy as np
from scipy.integrate import odeint

print("--- GROWTH RATE EVOLUTION (f*sigma8) TEST ---")

# ==========================================
# 1. OBSERVATIONAL DATA (BOSS DR12 + eBOSS)
# ==========================================
# Format: [Redshift z, f*sigma8, Error]
data_rsd = np.array([
    [0.38, 0.448, 0.038],  # BOSS Low-z
    [0.51, 0.455, 0.038],  # BOSS Mid-z
    [0.61, 0.410, 0.034],  # BOSS High-z (The "Dip")
    [1.48, 0.382, 0.026]   # eBOSS Quasars (High-z Check)
])

# Planck 2018 Baseline
SIGMA8_0_LCDM = 0.811
OM = 0.315

# ==========================================
# 2. PHYSICS PARAMETERS (Vacuum Elastodynamics)
# ==========================================
ETA_DRAG = 0.17    # Viscosity
Z_TRANS = 0.65     # Phase Transition
WIDTH = 0.15

def sigmoid(z):
    return 1.0 / (1.0 + np.exp((z - Z_TRANS)/WIDTH))

def hubble_E(a):
    z = 1.0/a - 1.0
    return np.sqrt(OM*(1+z)**3 + (1-OM))

def growth_ode(y, a, model='lcdm'):
    delta, delta_prime = y
    z = 1.0/a - 1.0
    E = hubble_E(a)
    
    # Friction Term
    dE_da = -1.5 * OM * (a**-4) / E
    friction = 3.0/a + dE_da/E
    
    # Viscosity (Vacuum Model Only)
    if model == 'viscous':
        visc_eff = ETA_DRAG * sigmoid(z)
        friction += visc_eff / a
        
    # Source Term
    source = 1.5 * OM / (a**5 * E**2)
    
    return [delta_prime, -friction * delta_prime + source * delta]

# ==========================================
# 3. RUN SIMULATION
# ==========================================
# Solve from z=100 to z=0
z_start = 100.0
a_grid = np.linspace(1.0/(1+z_start), 1.0, 500)
y0 = [a_grid[0], 1.0] # Initial condition (Matter Domination)

# Run LCDM
sol_lcdm = odeint(growth_ode, y0, a_grid, args=('lcdm',))
delta_lcdm = sol_lcdm[:, 0]
d_delta_lcdm = sol_lcdm[:, 1]

# Run Vacuum Model
sol_vac = odeint(growth_ode, y0, a_grid, args=('viscous',))
delta_vac = sol_vac[:, 0]
d_delta_vac = sol_vac[:, 1]

# ==========================================
# 4. CALCULATE OBSERVABLES
# ==========================================
z_axis = 1.0/a_grid - 1.0

# Growth Rate f = dln(delta)/dln(a)
f_lcdm = (a_grid / delta_lcdm) * d_delta_lcdm
f_vac  = (a_grid / delta_vac) * d_delta_vac

# Amplitude Normalization (Sigma8)
# Normalized to match at high-z (Early Universe Physics is identical)
# sig8(z) = sigma8_0 * D(z)/D(0)
# We anchor LCDM to Planck best-fit at z=0
sig8_lcdm = SIGMA8_0_LCDM * (delta_lcdm / delta_lcdm[-1])

# We anchor Vacuum to match LCDM amplitude at high-z (z=100)
# This ensures we don't break the CMB fit.
norm_factor = (sig8_lcdm[0] / delta_vac[0]) * delta_vac # Scale curve
sig8_vac = SIGMA8_0_LCDM * (delta_vac / delta_lcdm[-1]) # Simplified relative scaling

# Combine: f * sigma8
fs8_lcdm_curve = f_lcdm * sig8_lcdm
fs8_vac_curve  = f_vac * sig8_vac

# ==========================================
# 5. GENERATE REPORT
# ==========================================
print(f"\n{'Redshift':<10} | {'Data (+/- Err)':<20} | {'LCDM':<10} | {'Vacuum':<10} | {'Delta_Chi2'}")
print("-" * 75)

total_chi2_lcdm = 0
total_chi2_vac = 0

for row in data_rsd:
    z_target = row[0]
    val_data = row[1]
    err_data = row[2]
    
    # Interpolate Model Predictions to exact redshift
    pred_lcdm = np.interp(z_target, np.flip(z_axis), np.flip(fs8_lcdm_curve))
    pred_vac  = np.interp(z_target, np.flip(z_axis), np.flip(fs8_vac_curve))
    
    # Calculate Chi2 Contribution
    chi2_lcdm = ((pred_lcdm - val_data)/err_data)**2
    chi2_vac  = ((pred_vac - val_data)/err_data)**2
    
    total_chi2_lcdm += chi2_lcdm
    total_chi2_vac += chi2_vac
    
    # Improvement Metric
    d_chi2 = chi2_lcdm - chi2_vac
    status = f"{d_chi2:+.2f}"
    
    print(f"{z_target:<10} | {val_data:.3f} +/- {err_data:.3f}    | {pred_lcdm:.3f}      | {pred_vac:.3f}      | {status}")

print("-" * 75)
print(f"Total Chi2 (LCDM):   {total_chi2_lcdm:.2f}")
print(f"Total Chi2 (Vacuum): {total_chi2_vac:.2f}")
print(f"Improvement:         {total_chi2_lcdm - total_chi2_vac:.2f} ({(total_chi2_lcdm - total_chi2_vac)/total_chi2_lcdm * 100:.1f}%)")

if total_chi2_vac < total_chi2_lcdm:
    print("\nVERDICT: PASS. The Vacuum Model fits the growth history better.")
else:
    print("\nVERDICT: FAIL. The Vacuum Model makes the fit worse.")
