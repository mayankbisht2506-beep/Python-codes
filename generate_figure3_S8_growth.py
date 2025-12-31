import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# ==========================================
# 1. PHYSICAL CONSTANTS
# ==========================================
H0_PLANCK = 67.4
OM_PLANCK = 0.315
S8_PLANCK = 0.832

# Observational Targets (What we WANT)
S8_DES_TARGET = 0.776 

# --- SCIENTIFIC INPUTS (What we HAVE) ---
# We use the Observed Proton Load (0.16), NOT the theoretical limit (0.21).
# Source: Section 7.4.2
ETA_PHYSICAL = 0.1569    # The Honest Value
ETA_PEAK     = 0.31    # The Jamming Spike (Geometric Constant)
Z_TRANS      = 0.65    # Percolation Threshold

# ==========================================
# 2. S8 ENGINE
# ==========================================
def get_viscosity_profile(z):
    width = 0.1
    arg = (z - Z_TRANS) / width
    late_trigger = 1.0 / (1.0 + np.exp(arg))
    
    # Base Floor using PHYSICAL value
    base = ETA_PHYSICAL * late_trigger
    
    # Jamming Spike
    spike = (ETA_PEAK - ETA_PHYSICAL) * np.exp(-0.5 * ((z - Z_TRANS)/0.15)**2)
    
    return base + spike

def growth_ode(y, a, model='lcdm'):
    delta, delta_prime = y
    z = 1.0/a - 1.0
    
    E = np.sqrt(OM_PLANCK*(1+z)**3 + (1-OM_PLANCK))
    dE_da = -1.5 * OM_PLANCK * (a**-4) / E
    hubble_friction = 3.0/a + dE_da/E
    gravity_source = 1.5 * OM_PLANCK / (a**5 * E**2)
    
    if model == 'viscous':
        eta = get_viscosity_profile(z)
        friction_term = hubble_friction * (1.0 + eta)**2.0
    else:
        friction_term = hubble_friction
        
    return [delta_prime, -friction_term*delta_prime + gravity_source*delta]

# ==========================================
# 3. EXECUTION
# ==========================================
print(f"--- SCIENTIFIC VALIDATION: PROTON LOAD LIMIT ---")
print(f"Input Parameter: eta = {ETA_PHYSICAL} (Observed Proton Load)")

a_range = np.linspace(0.001, 1.0, 500)
y0 = [a_range[0], 1.0]

sol_lcdm = odeint(growth_ode, y0, a_range, args=('lcdm',))
sol_visc = odeint(growth_ode, y0, a_range, args=('viscous',))

growth_suppression = sol_visc[-1,0] / sol_lcdm[-1,0]
s8_pred = S8_PLANCK * growth_suppression

print(f"\nRESULTS:")
print(f"   Standard LCDM S8:    {S8_PLANCK:.3f}")
print(f"   Target (Weak Lens):  {S8_DES_TARGET:.3f}")
print(f"   Vacuum Prediction:   {s8_pred:.3f}")

print(f"\nSCIENTIFIC VERDICT:")
if abs(s8_pred - S8_DES_TARGET) < 0.02:
    print(f"   FULL RESOLUTION (Matches Data)")
elif s8_pred < S8_PLANCK:
    print(f"   PARTIAL RESOLUTION (Alleviates Tension, but gap remains)")
else:
    print(f"   FAILURE (No improvement)")

# Visualization
plt.figure(figsize=(10,5))
z_axis = 1/a_range - 1
plt.plot(z_axis, sol_lcdm[:,0]/sol_lcdm[-1,0], 'k--', label='LCDM (0.832)')
plt.plot(z_axis, sol_visc[:,0]/sol_lcdm[-1,0], 'b-', linewidth=2, label=f'Honest Vacuum (S8={s8_pred:.3f})')
plt.axhline(S8_DES_TARGET/S8_PLANCK, color='g', linestyle=':', label='Observational Target')
plt.xlabel('Redshift z')
plt.ylabel('Growth Factor Ratio')
plt.title(f'Scientific Prediction: Proton Load (eta={ETA_PHYSICAL})')
plt.legend()
plt.gca().invert_xaxis()
plt.show()
