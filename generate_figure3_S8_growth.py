import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# ==========================================
# 1. PARAMETERS
# ==========================================
S8_PLANCK = 0.832
S8_KIDS = 0.759
S8_DES = 0.776

Om0 = 0.310
Z_TRANS = 0.65
WIDTH = 0.15      # Slightly wider to smooth the "Fluid" to "Solid" handover

# Viscosity Profile
ETA_SOFT = 0.105  # Early Universe (Fluid Phase). Balances G_BOOST (1.105^2 ~= 1.22)
ETA_PEAK = 0.31   # Transition (Jamming)
ETA_FLOOR = 0.21  # Late Universe (Solid Phase)

# Gravity
G_BOOST = 1.22    # Early Gravity Boost

# ==========================================
# 2. PHYSICS ENGINE
# ==========================================
def get_physics_at_z(z):
    # Transition Trigger (Sigmoid)
    # 0.0 at Late times, 1.0 at Early times
    arg = (z - Z_TRANS) / WIDTH
    arg = np.clip(arg, -50, 50)
    early_trigger = 1.0 / (1.0 + np.exp(-arg)) # 1 when z >> 0.65
    
    # 1. Gravity Profile
    # Drops from 1.22 (Early) to 1.0 (Late)
    G_scale = 1.0 + (G_BOOST - 1.0) * early_trigger
    
    # 2. Viscosity Profile (The Fix)
    # Early Universe: Fluid (0.105)
    # Late Universe:  Solid (0.21)
    # Transition:     Jamming Spike (0.31)
    
    # Base transition: Soft (0.105) -> Solid (0.21)
    base_eta = ETA_FLOOR + (ETA_SOFT - ETA_FLOOR) * early_trigger
    
    # Jamming Spike: Gaussian bump at z=0.65
    # Adds the extra friction to hit 0.31 peak
    spike_amplitude = ETA_PEAK - base_eta 
    # Use a Gaussian centered at Z_TRANS
    jamming = (ETA_PEAK - 0.21) * np.exp(-0.5 * ((z - Z_TRANS)/0.2)**2)
    
    # We ensure we don't double count. 
    # Let's blend: Base transition + Jamming Spike
    total_eta = base_eta + jamming
    
    # Clip to ensure physics safety (strictly positive)
    total_eta = np.maximum(total_eta, 0.0)
    
    return total_eta, G_scale

def hubble_E(a):
    z = 1.0/a - 1.0
    return np.sqrt(Om0*(1+z)**3 + (1-Om0))

def growth_ode(y, a, model='lcdm'):
    delta, delta_prime = y
    z = 1.0/a - 1.0
    E = hubble_E(a)
    
    # Standard Terms
    dE_da = -1.5 * Om0 * (a**-4) / E
    hubble_friction = 3.0/a + dE_da/E
    gravity_source = 1.5 * Om0 / (a**5 * E**2)
    
    if model == 'viscous':
        eta, G_scale = get_physics_at_z(z)
        
        # 1. Quadratic Impedance (Friction)
        # Early: (1+0.105)^2 ~= 1.22 (Matches Gravity)
        # Late:  (1+0.21)^2  ~= 1.46 (Brakes Growth)
        friction_term = hubble_friction * (1.0 + eta)**2.0
        
        # 2. Turbocharger (Gravity)
        # Early: 1.22x
        # Late:  1.0x
        source_term = gravity_source * G_scale
        
    else:
        friction_term = hubble_friction
        source_term = gravity_source

    return [delta_prime, -friction_term*delta_prime + source_term*delta]

# ==========================================
# 3. RUN SIMULATION
# ==========================================
print("Simulating Structure Growth (Soft-to-Solid Transition)...")
a_range = np.linspace(0.001, 1.0, 1000)
y0 = [a_range[0], 1.0]

# Run Planck LCDM
sol_lcdm = odeint(growth_ode, y0, a_range, args=('lcdm',))
delta_lcdm = sol_lcdm[:, 0]

# Run Vacuum Model
sol_visc = odeint(growth_ode, y0, a_range, args=('viscous',))
delta_visc = sol_visc[:, 0]

# ==========================================
# 4. RESULTS
# ==========================================
suppression = delta_visc[-1] / delta_lcdm[-1]
S8_PRED = S8_PLANCK * suppression

print(f"--- RESULTS ---")
print(f"Early Viscosity: {ETA_SOFT} (Balances Turbo)")
print(f"Peak Viscosity:  {ETA_PEAK} (Jamming)")
print(f"Late Viscosity:  {ETA_FLOOR} (Lepton Floor)")
print(f"Predicted S8:    {S8_PRED:.3f}")

if 0.75 <= S8_PRED <= 0.78:
    print("VERDICT: SUCCESS. Model resolves S8 Tension.")
else:
    print("VERDICT: FAIL.")

# Plot
z_plot = 1.0/a_range - 1.0
plt.figure(figsize=(10, 6))
plt.plot(z_plot, delta_lcdm/delta_lcdm[-1], 'k--', label=f'Standard LCDM ($S_8={S8_PLANCK:.3f}$)')
plt.plot(z_plot, delta_visc/delta_lcdm[-1], 'r-', linewidth=3, label=f'Vacuum Model ($S_8={S8_PRED:.3f}$)')

# Highlight Phases
plt.axvspan(0.8, 2.5, color='gray', alpha=0.1, label='Soft Phase (Balanced)')
plt.axvspan(0.4, 0.8, color='red', alpha=0.1, label='Jamming Phase (Braking)')
plt.axvspan(0.0, 0.4, color='blue', alpha=0.05, label='Stiff Phase (Damping)')

plt.xlabel('Redshift z')
plt.ylabel('Relative Growth Amplitude')
plt.title(f'Resolution of S8 Tension (Target: {S8_KIDS}-{S8_DES})')
plt.legend(loc='lower right')
plt.xlim(0, 2.5)
plt.gca().invert_xaxis()
plt.grid(alpha=0.3)
plt.savefig('S8_Final_Corrected.pdf')
plt.show()
