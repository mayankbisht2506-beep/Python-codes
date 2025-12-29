import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. PARAMETERS (Quadruple Concordance)
# ==========================================
H0_PLANCK = 67.4 
Om_PLANCK = 0.315

# TARGET: The "Stiff" Vacuum H0
# Reduced from 74.0 to 73.4 due to Lepton Viscosity Drag
H0_TARGET = 73.4    
Z_TRANS = 0.65      # Percolation Threshold
WIDTH = 0.1         # Transition Width

# ==========================================
# 2. PHYSICS ENGINE
# ==========================================
def get_hubble_evolution(z_array):
    # Standard LCDM Background
    E_z = np.sqrt(Om_PLANCK * (1 + z_array)**3 + (1 - Om_PLANCK))
    
    # VACUUM PHASE TRANSITION (CORRECTED SIGMOID)
    # We want the boost to be ON at Low z (z < 0.65) and OFF at High z.
    # Logic:
    # If z=0:   arg = (0 - 0.65)/0.1 = -6.5.  exp(-6.5) ~ 0.   Sigmoid ~ 1. (Active)
    # If z=2:   arg = (2 - 0.65)/0.1 = 13.5.  exp(13.5) ~ Big. Sigmoid ~ 0. (Inactive)
    
    arg = (z_array - Z_TRANS) / WIDTH
    # Clamp to prevent overflow in exp (Standard safety)
    arg = np.clip(arg, -100, 100) 
    
    sigmoid = 1.0 / (1.0 + np.exp(arg))
    
    # Boost Amplitude (Scaling 67.4 -> 73.4)
    boost_factor = H0_TARGET / H0_PLANCK
    
    # Apply Boost to the Late Universe
    effective_H0_scaling = 1.0 + (boost_factor - 1.0) * sigmoid
    
    # Calculate H(z)
    H_vacuum = H0_PLANCK * E_z * effective_H0_scaling
    
    # Reference Models
    H_lcdm = H0_PLANCK * E_z
    H_shoes = 73.04 * np.sqrt(0.3 * (1 + z_array)**3 + 0.7)
    
    return H_vacuum, H_lcdm, H_shoes

# ==========================================
# 3. GENERATE & PLOT
# ==========================================
print("Simulating Vacuum Phase Transition (Fixed Direction)...")
z_eval = np.linspace(0, 2.5, 500)
H_vac, H_lcdm, H_shoes = get_hubble_evolution(z_eval)

print(f"--- TRANSITION DIAGNOSTICS ---")
print(f"H0 (Planck Base): {H_lcdm[0]:.2f} km/s/Mpc")
print(f"H0 (Vacuum):      {H_vac[0]:.2f} km/s/Mpc (Target: {H0_TARGET})")
print(f"Transition z:     {Z_TRANS}")
print(f"Check High-z:     Vacuum H(2.0)={H_vac[-1]:.1f} vs Planck H(2.0)={H_lcdm[-1]:.1f}")

if abs(H_vac[0] - H0_TARGET) < 0.1:
    print("VERDICT: SUCCESS. Phase Transition accurately boosts H0.")
else:
    print("VERDICT: FAIL. Boost logic error.")

# Plotting
plt.figure(figsize=(10, 6))

# Plot Lines
plt.plot(z_eval, H_lcdm, 'k--', label='Planck 2018 ($H_0=67.4$)')
plt.plot(z_eval, H_shoes, 'g:', linewidth=2, label='SH0ES ($H_0=73.0$)')
plt.plot(z_eval, H_vac, 'r-', linewidth=3, label=f'Vacuum Model ($H_0={H_vac[0]:.1f}$)')

# Highlight the Transition Zone
plt.axvspan(Z_TRANS - WIDTH, Z_TRANS + WIDTH, color='red', alpha=0.1, label='Transition Zone')
plt.axvline(Z_TRANS, color='red', linestyle=':', alpha=0.5)

plt.xlabel('Redshift $z$', fontsize=12)
plt.ylabel('$H(z)$ [km/s/Mpc]', fontsize=12)
plt.title(r'Figure 2: The "Stiff" Vacuum Transition ($z \approx 0.65$)', fontsize=14)
plt.legend(loc='upper left')
plt.xlim(0, 2.0)
plt.ylim(60, 200)
plt.grid(alpha=0.3)

plt.savefig('Figure2_Hubble_Transition.pdf')
plt.show()
