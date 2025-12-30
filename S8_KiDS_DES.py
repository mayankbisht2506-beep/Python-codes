import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# ==========================================
# 1. PARAMETERS (Corrected to Add 36)
# ==========================================
S8_PLANCK = 0.832
Om0 = 0.310

# PHYSICS INPUTS (Geometric Unity)
# No arbitrary "7.4" scaling factor allowed.
ETA_FLOOR = 0.21       # Lepton Sum Rule (Section 7.4)
ETA_PEAK = 0.31        # Percolation Threshold (Section 7.5)
Z_TRANS = 0.65         # Transition Redshift
WIDTH = 0.1            # Phase Transition Width

# DATA TARGETS
S8_KIDS = 0.766
ERR_KIDS = 0.020
S8_DES = 0.776
ERR_DES = 0.017

# ==========================================
# 2. PHYSICS ENGINE (Quadratic Impedance)
# ==========================================
def get_viscosity(z):
    """
    Combines the Late-Time Stiffness Floor (0.21) 
    with the Jamming Transition Spike (0.31).
    """
    # 1. The Floor (Sigmoid activation)
    arg = (z - Z_TRANS) / WIDTH
    # Avoid overflow in exp
    late_trigger = np.where(arg > 50, 0.0, 1.0 / (1.0 + np.exp(arg)))
    base_viscosity = ETA_FLOOR * late_trigger
    
    # 2. The Jamming Spike (Gaussian at z=0.65)
    # Represents the 'clumping' at the percolation threshold
    spike = (ETA_PEAK - ETA_FLOOR) * np.exp(-0.5 * ((z - Z_TRANS)/0.15)**2)
    
    return base_viscosity + spike

def hubble_E(a):
    z = 1.0/a - 1.0
    return np.sqrt(Om0*(1+z)**3 + (1-Om0))

def growth_ode(y, a, model='lcdm'):
    delta, delta_prime = y
    z = 1.0/a - 1.0
    E = hubble_E(a)
    
    # Standard Cosmological Friction & Source
    dE_da = -1.5 * Om0 * (a**-4) / E
    hubble_friction = 3.0/a + dE_da/E
    source = 1.5 * Om0 / (a**5 * E**2)
    
    if model == 'viscous':
        eta = get_viscosity(z)
        
        # --- RECTIFICATION START ---
        # OLD (Wrong): Linear Scaling with arbitrary 7.4 factor
        # friction += (7.4 * eta) / a
        
        # NEW (Correct): Quadratic Impedance (Eq. 89 in Paper)
        # Friction scales as the square of the defect density (1+eta)^2
        friction_term = hubble_friction * (1.0 + eta)**2.0
        # --- RECTIFICATION END ---
        
        # We assume standard gravity source (Cancellation Assumption)
        # to isolate the suppression effect.
        return [delta_prime, -friction_term*delta_prime + source*delta]
        
    else:
        # Standard LCDM Friction
        return [delta_prime, -hubble_friction*delta_prime + source*delta]

# ==========================================
# 3. RUN SIMULATION
# ==========================================
print("Running Geometric Unity Simulation...")
a_range = np.linspace(0.001, 1.0, 1000)
y0 = [a_range[0], 1.0]

# Run Models
sol_lcdm = odeint(growth_ode, y0, a_range, args=('lcdm',))
sol_visc = odeint(growth_ode, y0, a_range, args=('viscous',))

# Results
s8_pred = S8_PLANCK * (sol_visc[-1, 0] / sol_lcdm[-1, 0])

print(f"\n--- S8 RESOLUTION RESULT ---")
print(f"Physics Model:   Quadratic Impedance (n=2)")
print(f"Viscosity Inputs: Floor={ETA_FLOOR}, Peak={ETA_PEAK}")
print(f"Predicted S8:    {s8_pred:.3f}")
print(f"Target (KiDS):   {S8_KIDS} +/- {ERR_KIDS}")
print(f"Target (DES):    {S8_DES} +/- {ERR_DES}")

if 0.76 <= s8_pred <= 0.78:
    print("VERDICT: SUCCESS. Matches DES/KiDS without arbitrary scaling.")
else:
    print("VERDICT: CHECK PARAMETERS.")

# ==========================================
# 4. PLOTTING
# ==========================================
plt.figure(figsize=(9,6))

# Define Bars
labels = ['Planck 2018', 'KiDS-1000', 'DES Y3', 'Vacuum Model\n(This Paper)']
values = [S8_PLANCK, S8_KIDS, S8_DES, s8_pred]
errors = [0.013, ERR_KIDS, ERR_DES, 0.013] # Assume Planck-like error for model
colors = ['black', 'blue', 'green', 'firebrick']

# Plot Points with Error Bars
for i in range(4):
    plt.errorbar(i, values[i], yerr=errors[i], fmt='o', 
                 color=colors[i], capsize=5, markersize=8, label=labels[i])
    # Add faint bar for visual weight
    plt.bar(i, values[i], width=0.4, color=colors[i], alpha=0.1)

# Add Target Band (The "Truth" Zone)
plt.axhspan(S8_KIDS-ERR_KIDS, S8_DES+ERR_DES, color='gray', alpha=0.1, label='Concordance Zone')

plt.xticks(range(4), labels)
plt.ylabel(r'$S_8$ Amplitude')
plt.title(f'Resolution of Clustering Tension\nQuadratic Impedance Model ($S_8 \\approx {s8_pred:.3f}$)')
plt.ylim(0.70, 0.86)
plt.grid(axis='y', alpha=0.3)
plt.legend(loc='upper right')

plt.tight_layout()
plt.savefig('Figure3_S8_Rectified.png', dpi=300)
plt.show()
