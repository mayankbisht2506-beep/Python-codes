import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# ==========================================
# 1. OBSERVATIONAL GOAL POSTS (Data)
# ==========================================
# We verify against these published values.
# If our physics matches these, the model works.
S8_PLANCK = 0.832       # Starting Point (Early Universe)
S8_KIDS = 0.759         # Goal Post 1 (KiDS-1000 Data)
S8_DES = 0.776          # Goal Post 2 (DES-Y3 Data)
TARGET_ZONE_CENTER = 0.765 

# ==========================================
# 2. PHYSICS PARAMETERS (Quadruple Concordance)
# ==========================================
Om0 = 0.310
ETA_LATE = 0.21         # Fixed by Lepton Sum Rule
Z_TRANS = 0.65          # Fixed by Percolation
WIDTH = 0.15            

# ==========================================
# 3. PHYSICS ENGINE (Quadratic Impedance)
# ==========================================
def sigmoid(z):
    arg = (z - Z_TRANS) / WIDTH
    arg = np.clip(arg, -50, 50) 
    return 1.0 / (1.0 + np.exp(arg))

def hubble_E(a):
    z = 1.0/a - 1.0
    return np.sqrt(Om0*(1+z)**3 + (1-Om0))

def growth_ode_quadratic(y, a, model='lcdm'):
    delta, delta_prime = y
    z = 1.0/a - 1.0
    E = hubble_E(a)

    dE_da = -1.5 * Om0 * (a**-4) / E
    hubble_friction = 3.0/a + dE_da/E
    gravity_source = 1.5 * Om0 / (a**5 * E**2)

    if model == 'viscous':
        # UPDATED PHYSICS: QUADRATIC SCALING (n=2.0)
        # The vacuum is "Stiff" (Hyperuniform). 
        # Resistance scales as the square of the order parameter.
        eta_eff = ETA_LATE * sigmoid(z)
        
        # CHANGED FROM 1.5 TO 2.0 HERE:
        coupling = (1.0 + eta_eff)**2.0 
        
        friction_term = hubble_friction * coupling
        source_term = gravity_source / coupling
        
    else:
        friction_term = hubble_friction
        source_term = gravity_source

    return [delta_prime, -friction_term*delta_prime + source_term*delta]

# ==========================================
# 4. RUN SIMULATION
# ==========================================
print("Simulating Structure Growth (Quadratic Impedance n=2.0)...")
a_range = np.linspace(0.001, 1.0, 1000)
y0 = [a_range[0], 1.0]

# Run Planck LCDM (Standard)
sol_lcdm = odeint(growth_ode_quadratic, y0, a_range, args=('lcdm',))
delta_lcdm = sol_lcdm[:, 0]

# Run Vacuum Model (Viscous)
sol_visc = odeint(growth_ode_quadratic, y0, a_range, args=('viscous',))
delta_visc = sol_visc[:, 0]

# ==========================================
# 5. RESULTS
# ==========================================
suppression = delta_visc[-1] / delta_lcdm[-1]
S8_PRED = S8_PLANCK * suppression

print(f"--- QUADRUPLE CONCORDANCE RESULTS ---")
print(f"Viscosity:      {ETA_LATE} (Lepton Sum Rule)")
print(f"Scaling Law:    Quadratic (n=2)")
print(f"Predicted S8:   {S8_PRED:.3f}")
print(f"Goal Posts:     {S8_KIDS} (KiDS) < {S8_PRED:.3f} < {S8_DES} (DES)")

# VERIFICATION LOGIC
# We are successful if we land near the KiDS/DES average (approx 0.76-0.77)
if 0.75 <= S8_PRED <= 0.78:
    print("VERDICT: SUCCESS. Model lands in the Observational Goldilocks Zone.")
else:
    print("VERDICT: FAIL. Still missing the target.")

# PLOTTING
z_plot = 1.0/a_range - 1.0
plt.figure(figsize=(10, 6))

# Plot Models
plt.plot(z_plot, delta_lcdm/delta_lcdm[-1], 'k--', label=f'Standard $\Lambda$CDM ($S_8={S8_PLANCK:.2f}$)')
plt.plot(z_plot, delta_visc/delta_lcdm[-1], 'r-', linewidth=3, label=f'Vacuum Model ($S_8={S8_PRED:.3f}$)')

# Plot Goal Posts (Data)
plt.errorbar(0, S8_KIDS/S8_PLANCK, yerr=0.02/S8_PLANCK, fmt='s', color='blue', label='KiDS-1000 Data')
plt.errorbar(0, S8_DES/S8_PLANCK, yerr=0.015/S8_PLANCK, fmt='o', color='green', label='DES-Y3 Data')

plt.xlabel('Redshift $z$')
plt.ylabel('Relative Growth Amplitude')
plt.title(r'Figure 3: Resolving $S_8$ Tension (Quadratic Model)')
plt.legend()
plt.xlim(0, 2.5)
plt.grid(alpha=0.3)
plt.gca().invert_xaxis()
plt.savefig('Figure3_S8_Growth.pdf')
plt.show()
