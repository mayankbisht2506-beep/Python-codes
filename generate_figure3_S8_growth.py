import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# ==========================================
# 1. PARAMETERS (Final Production)
# ==========================================
S8_PLANCK = 0.832
S8_KIDS = 0.759
S8_DES = 0.776

Om0 = 0.310
ETA_FLOOR = 0.21    # Lepton Stiffness (Late Time)
ETA_PEAK = 0.31     # Percolation Threshold (Transition Spike)
Z_TRANS = 0.65      # Transition Redshift
WIDTH = 0.1

# ==========================================
# 2. PHYSICS ENGINE (Effective Model)
# ==========================================
def get_effective_viscosity(z):
    # 1. Late-Time Activation (Sigmoid)
    # The vacuum becomes viscous only as it crystallizes.
    arg = (z - Z_TRANS) / WIDTH
    # 0 at high z, 1 at low z
    late_trigger = 1.0 / (1.0 + np.exp(arg)) 
    
    # 2. The Physics Profile
    # Base: Settle to 0.21
    base_viscosity = ETA_FLOOR * late_trigger
    
    # Peak: Spike to 0.31 during the transition
    # Gaussian centered at Z_TRANS
    # Amplitude is difference (0.31 - 0.21 = 0.10)
    spike_amplitude = ETA_PEAK - ETA_FLOOR
    jamming_spike = spike_amplitude * np.exp(-0.5 * ((z - Z_TRANS)/0.15)**2)
    
    # Total Effective Viscosity
    # This combines the floor and the peak
    eta_eff = base_viscosity + jamming_spike
    
    return eta_eff

def growth_ode_effective(y, a, model='lcdm'):
    delta, delta_prime = y
    z = 1.0/a - 1.0
    
    E = np.sqrt(Om0*(1+z)**3 + (1-Om0))
    dE_da = -1.5 * Om0 * (a**-4) / E
    hubble_friction = 3.0/a + dE_da/E
    gravity_source = 1.5 * Om0 / (a**5 * E**2)
    
    if model == 'viscous':
        eta = get_effective_viscosity(z)
        
        # QUADRATIC IMPEDANCE (Eq 89)
        # We apply the full profile (Peak + Floor)
        friction_term = hubble_friction * (1.0 + eta)**2.0
        
        # Cancellation Assumption:
        # We assume Early Universe Gravity (1.22x) cancels Early Soft Viscosity.
        # So we use standard gravity here.
        source_term = gravity_source 
        
    else:
        friction_term = hubble_friction
        source_term = gravity_source

    return [delta_prime, -friction_term*delta_prime + source_term*delta]

# ==========================================
# 3. RUN SIMULATION
# ==========================================
print("Simulating Structure Growth (Effective Model with Peak)...")
a_range = np.linspace(0.001, 1.0, 1000)
y0 = [a_range[0], 1.0]

# Run Planck LCDM
sol_lcdm = odeint(growth_ode_effective, y0, a_range, args=('lcdm',))

# Run Vacuum Model (With Jamming Peak)
sol_visc = odeint(growth_ode_effective, y0, a_range, args=('viscous',))

# ==========================================
# 4. RESULTS
# ==========================================
S8_PRED = S8_PLANCK * (sol_visc[-1, 0] / sol_lcdm[-1, 0])

print(f"--- RESULTS ---")
print(f"Viscosity Profile: Floor={ETA_FLOOR} -> Peak={ETA_PEAK}")
print(f"Predicted S8:      {S8_PRED:.3f}")
print(f"Target Range:      {S8_KIDS} (KiDS) - {S8_DES} (DES)")

if 0.76 <= S8_PRED <= 0.78:
    print("VERDICT: SUCCESS. Perfect match with DES-Y3.")
else:
    print("VERDICT: CHECK PARAMETERS.")

# Plot
z_plot = 1.0/a_range - 1.0
plt.figure(figsize=(10, 6))
plt.plot(z_plot, sol_lcdm[:,0]/sol_lcdm[-1,0], 'k--', label=f'Standard LCDM ($S_8={S8_PLANCK:.3f}$)')
plt.plot(z_plot, sol_visc[:,0]/sol_lcdm[-1,0], 'r-', linewidth=3, label=f'Vacuum Model ($S_8={S8_PRED:.3f}$)')

plt.axvspan(0.5, 0.8, color='red', alpha=0.1, label='Jamming Phase ($z \\approx 0.65$)')
plt.xlabel('Redshift z')
plt.ylabel('Relative Growth')
plt.title(f'Resolution of S8 Tension (with Jamming Peak)')
plt.legend()
plt.gca().invert_xaxis()
plt.grid(alpha=0.3)
plt.show()
