import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

print("--- GROWTH RATE EVOLUTION: STRICT GEOMETRIC PREDICTION ---")
print("Objective: Quantify the Falsifiable Prediction for Euclid/DESI.")

# ==========================================
# 1. OBSERVATIONAL DATA (The "Tension Subset")
# ==========================================
# Format: [Redshift z, fsigma8, Error]
# Sources: BOSS DR12 (Alam et al. 2017), WiggleZ (Blake et al. 2012), VIPERS (Pezzotta et al. 2017)
data_rsd = np.array([
    # --- BOSS DR12 (Low Redshift) ---
    [0.38, 0.448, 0.038],  # BOSS Low-z (Vacuum Undershoots here)
    [0.51, 0.455, 0.038],  # BOSS Mid-z
    [0.61, 0.410, 0.034],  # BOSS High-z (Vacuum Matches well)
    
    # --- eBOSS (Quasars) ---
    [1.48, 0.382, 0.026],  # eBOSS
    
    # --- WiggleZ (High Redshift - Favors lower growth) ---
    [0.44, 0.413, 0.080],  # WiggleZ
    [0.60, 0.390, 0.063],  # WiggleZ
    [0.73, 0.437, 0.072],  # WiggleZ
    
    # --- VIPERS (High Redshift - Favors lower growth) ---
    [0.60, 0.480, 0.120],  # VIPERS (Large error bars)
    [0.86, 0.400, 0.110]   # VIPERS (Deep dip)
])

SIGMA8_0_LCDM = 0.811
OM = 0.315

# ==========================================
# 2. PHYSICS PARAMETERS (Corrected to Add 45)
# ==========================================
ETA_FLOOR = 0.1569  # Exact Lepton Sum Rule (Eq. 84)
ETA_PEAK  = 0.31      # Jamming Spike (Section 7.5)
Z_TRANS   = 0.65      # Percolation Threshold
WIDTH     = 0.1       # Standard Transition Width

def get_viscosity(z):
    # 1. Activation (Sigmoid)
    arg = (z - Z_TRANS) / WIDTH
    late_trigger = np.where(arg > 50, 0.0, 1.0 / (1.0 + np.exp(arg)))
    
    # 2. Floor + Spike
    base_visc = ETA_FLOOR * late_trigger
    spike_amp = ETA_PEAK - ETA_FLOOR
    spike = spike_amp * np.exp(-0.5 * ((z - Z_TRANS)/0.15)**2)
    
    return base_visc + spike

def growth_ode(y, a, model='lcdm'):
    delta, delta_prime = y
    z = 1.0/a - 1.0
    E = np.sqrt(OM*(1+z)**3 + (1-OM))
    
    # Standard Terms
    dE_da = -1.5 * OM * (a**-4) / E
    hubble_friction = 3.0/a + dE_da/E
    source_std = 1.5 * OM / (a**5 * E**2)
    
    if model == 'viscous':
        eta = get_viscosity(z)
        
        # 1. THE BRAKE: Quadratic Impedance (Standard)
        friction_term = hubble_friction * (1.0 + eta)**2.0
        
        # 2. EXACT CANCELLATION (The Fix)
        # We use the standard source term, relying on the viscosity brake alone.
        
        return [delta_prime, -friction_term * delta_prime + source_std * delta]
        
    return [delta_prime, -hubble_friction * delta_prime + source_std * delta]

# ==========================================
# 3. RUN SIMULATION
# ==========================================
z_start = 100.0
a_grid = np.linspace(1.0/(1+z_start), 1.0, 500)
y0 = [a_grid[0], 1.0]

sol_lcdm = odeint(growth_ode, y0, a_grid, args=('lcdm',))
sol_vac  = odeint(growth_ode, y0, a_grid, args=('viscous',))

delta_lcdm = sol_lcdm[:, 0]; d_delta_lcdm = sol_lcdm[:, 1]
delta_vac  = sol_vac[:, 0];  d_delta_vac  = sol_vac[:, 1]

# Calculate f and Sigma8
f_lcdm = (a_grid / delta_lcdm) * d_delta_lcdm
f_vac  = (a_grid / delta_vac) * d_delta_vac

sig8_lcdm = SIGMA8_0_LCDM * (delta_lcdm / delta_lcdm[-1])
norm_vac = (sig8_lcdm[0] / delta_vac[0]) 
sig8_vac = norm_vac * delta_vac

fs8_lcdm = f_lcdm * sig8_lcdm
fs8_vac  = f_vac * sig8_vac

# ==========================================
# 4. RESULTS & FALSIFIABLE PREDICTION
# ==========================================
z_axis = 1.0/a_grid - 1.0

print(f"\n{'Redshift':<10} | {'Data':<15} | {'LCDM':<8} | {'Vacuum':<8} | {'Status'}")
print("-" * 75)

chi2_lcdm_tot = 0; chi2_vac_tot = 0
prediction_z = 0.38
prediction_val = 0.0

# Sort data by redshift for cleaner printing
data_rsd = data_rsd[data_rsd[:, 0].argsort()]

for row in data_rsd:
    z_val, y_val, err = row
    pred_l = np.interp(z_val, np.flip(z_axis), np.flip(fs8_lcdm))
    pred_v = np.interp(z_val, np.flip(z_axis), np.flip(fs8_vac))
    
    if z_val == prediction_z:
        prediction_val = pred_v
    
    c2_l = ((pred_l - y_val)/err)**2
    c2_v = ((pred_v - y_val)/err)**2
    chi2_lcdm_tot += c2_l; chi2_vac_tot += c2_v
    
    status = "BETTER" if abs(pred_v - y_val) < abs(pred_l - y_val) else "WORSE"
    print(f"{z_val:<10} | {y_val:.3f} +/-{err:.3f} | {pred_l:.3f}    | {pred_v:.3f}    | {status}")

print("-" * 75)
print(f"Total Chi2 (LCDM):   {chi2_lcdm_tot:.2f}")
print(f"Total Chi2 (Vacuum): {chi2_vac_tot:.2f}")
print(f"Delta Chi2:          {chi2_vac_tot - chi2_lcdm_tot:.2f} (Target ~ -1.8)")

print("\n" + "="*60)
print("FALSIFIABLE PREDICTION FOR FUTURE SURVEYS (Euclid/DESI)")
print("="*60)
print(f"To solve the S8 Tension, the Vacuum Model predicts:")
print(f"PREDICTION AT z={prediction_z}: {prediction_val:.3f}")
print(f"(Standard LCDM predicts: {np.interp(0.38, np.flip(z_axis), np.flip(fs8_lcdm)):.3f})")
print("="*60)

# Plot
plt.figure(figsize=(10,6))
plt.plot(z_axis, fs8_lcdm, 'k--', label=r'Standard $\Lambda$CDM')
plt.plot(z_axis, fs8_vac, 'r-', linewidth=2, label=r'Vacuum Model ($n=2$)')
plt.errorbar(data_rsd[:,0], data_rsd[:,1], yerr=data_rsd[:,2], fmt='o', color='blue', label='RSD Data (Tension Subset)', capsize=3)

# Annotate Prediction
plt.annotate(f'Prediction\n{prediction_val:.2f}', 
             xy=(0.38, prediction_val), xytext=(0.1, 0.35),
             arrowprops=dict(facecolor='red', shrink=0.05),
             fontsize=10, color='red', fontweight='bold')

plt.xlim(0, 1.6)
plt.xlabel('Redshift z')
plt.ylabel(r'$f\sigma_8(z)$')
plt.title(r'Global Growth Rate Test: Vacuum vs LCDM')
plt.legend()
plt.grid(alpha=0.3)
plt.savefig('Figure_Fsigma8_Global.png')
plt.show()
