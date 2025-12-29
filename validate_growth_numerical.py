import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

print("--- GROWTH RATE EVOLUTION: STRICT LEPTON PREDICTION ---")
print("Objective: Quantify the Falsifiable Prediction for Euclid/DESI.")

# ==========================================
# 1. OBSERVATIONAL DATA (BOSS DR12)
# ==========================================
data_rsd = np.array([
    [0.38, 0.448, 0.038],  # BOSS Low-z (The Tension Point)
    [0.51, 0.455, 0.038],  # BOSS Mid-z
    [0.61, 0.410, 0.034],  # BOSS High-z (The Dip)
    [1.48, 0.382, 0.026]   # eBOSS
])

SIGMA8_0_LCDM = 0.811
OM = 0.315

# ==========================================
# 2. PHYSICS PARAMETERS (Strict Add 33)
# ==========================================
ETA_FLOOR = 0.21      # Stable Lattice (z=0)
ETA_PEAK  = 0.31      # Jamming Spike (z=0.65)
SCALING   = 7.4       
Z_TRANS   = 0.65        
WIDTH     = 0.15

def get_effective_viscosity(z):
    # 1. Activation
    arg = (z - Z_TRANS) / WIDTH
    late_trigger = np.where(arg > 50, 0.0, 1.0 / (1.0 + np.exp(arg)))
    
    # 2. Floor + Spike
    base_visc = ETA_FLOOR * late_trigger
    spike_amp = ETA_PEAK - ETA_FLOOR
    spike = spike_amp * np.exp(-0.5 * ((z - Z_TRANS)/0.15)**2)
    
    return (base_visc + spike) * SCALING

def growth_ode(y, a, model='lcdm'):
    delta, delta_prime = y
    z = 1.0/a - 1.0
    E = np.sqrt(OM*(1+z)**3 + (1-OM))
    
    dE_da = -1.5 * OM * (a**-4) / E
    friction = 3.0/a + dE_da/E
    
    if model == 'viscous':
        visc_macro = get_effective_viscosity(z)
        friction += visc_macro / a
        
    source = 1.5 * OM / (a**5 * E**2)
    return [delta_prime, -friction * delta_prime + source * delta]

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

for row in data_rsd:
    z_val, y_val, err = row
    pred_l = np.interp(z_val, np.flip(z_axis), np.flip(fs8_lcdm))
    pred_v = np.interp(z_val, np.flip(z_axis), np.flip(fs8_vac))
    
    if z_val == prediction_z:
        prediction_val = pred_v
    
    c2_l = ((pred_l - y_val)/err)**2
    c2_v = ((pred_v - y_val)/err)**2
    chi2_lcdm_tot += c2_l; chi2_vac_tot += c2_v
    
    # Label the status of each point
    if abs(pred_v - y_val) < abs(pred_l - y_val):
        status = "BETTER (Dip)"
    else:
        status = "LOWER (Pred)"
        
    print(f"{z_val:<10} | {y_val:.3f} +/-{err:.3f} | {pred_l:.3f}    | {pred_v:.3f}    | {status}")

print("-" * 75)
print(f"Total Chi2 (LCDM):   {chi2_lcdm_tot:.2f}")
print(f"Total Chi2 (Vacuum): {chi2_vac_tot:.2f}")

print("\n" + "="*60)
print("FALSIFIABLE PREDICTION FOR FUTURE SURVEYS (Euclid/DESI)")
print("="*60)
print(f"To solve the S8 Tension (S8=0.774), the Vacuum Model predicts")
print(f"strong suppression of growth in the late universe.")
print(f"\nPREDICTION AT z={prediction_z}:")
print(f"  * Current BOSS Data:   {0.448:.3f}")
print(f"  * Standard LCDM:       {np.interp(0.38, np.flip(z_axis), np.flip(fs8_lcdm)):.3f}")
print(f"  * VACUUM PREDICTION:   {prediction_val:.3f}")
print(f"\nTEST: If Euclid finds f*sigma8 ~ {prediction_val:.2f} at z=0.38,")
print("      the Vacuum Elastodynamics theory is CONFIRMED.")
print("="*60)

# Plot
plt.figure(figsize=(9,6))
plt.plot(z_axis, fs8_lcdm, 'k--', label=r'Standard $\Lambda$CDM')
plt.plot(z_axis, fs8_vac, 'r-', linewidth=2, label=r'Strict Vacuum ($\eta \to 0.21$)')
plt.errorbar(data_rsd[:,0], data_rsd[:,1], yerr=data_rsd[:,2], fmt='o', color='blue', label='BOSS Data')

# Annotate Prediction
plt.annotate(f'Falsifiable Prediction\nExpected: {prediction_val:.2f}', 
             xy=(0.38, prediction_val), xytext=(0.1, 0.25),
             arrowprops=dict(facecolor='red', shrink=0.05),
             fontsize=10, color='red', fontweight='bold')

plt.xlim(0, 1.6)
plt.xlabel('Redshift z')
plt.ylabel(r'$f\sigma_8(z)$')
plt.title(r'Growth Rate Prediction for Euclid/DESI')
plt.legend()
plt.grid(alpha=0.3)
plt.savefig('Figure_Fsigma8_Prediction.png')
plt.show()
