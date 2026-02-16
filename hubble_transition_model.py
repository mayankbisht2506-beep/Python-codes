import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. PARAMETERS (Corrected to match paper)
# ==========================================
H0_PLANCK = 67.4
Om_PLANCK = 0.315

# VACUUM DYNAMICS (Eq 88 & 89)
H_FAST = 74.5            # Early Superfluid Ceiling (Pre-Transition)
DELTA_OM = 0.0523        # Inertial Counter-Load Drag
Om_EFF = 0.367           # Effective Late-Time Density (0.315 + 0.0523)
H_OBS = 72.53            # Terminal Decelerated Velocity

Z_TRANS = 0.65           # Percolation Threshold
WIDTH = 0.1              # Transition Width (Sigmoidal Relaxation)

# ==========================================
# PHYSICS ENGINE (Actual Phase Transition)
# ==========================================
def get_hubble_evolution(z_array):
    # 1. Planck Baseline (Standard LCDM)
    E_z_planck = np.sqrt(Om_PLANCK * (1 + z_array)**3 + (1 - Om_PLANCK))
    H_lcdm = H0_PLANCK * E_z_planck

    # 2. SH0ES Reference
    H_shoes = 73.04 * np.sqrt(0.3 * (1 + z_array)**3 + 0.7)

    # 3. Vacuum Elastodynamics (Dynamic Phase Transition)
    # Uses a hyperbolic tangent to model the second-order relaxation at z=0.65
    # Late Universe (z < 0.65) -> transition_factor approaches 1
    # Early Universe (z > 0.65) -> transition_factor approaches 0
    transition_factor = 0.5 * (1 - np.tanh((z_array - Z_TRANS) / WIDTH))

    # Dynamically scale the parameters based on the epoch
    H_dynamic = H_OBS * transition_factor + H_FAST * (1 - transition_factor)
    Om_dynamic = Om_EFF * transition_factor + Om_PLANCK * (1 - transition_factor)

    # Calculate H(z) using the dynamically evolving metric
    E_z_vac = np.sqrt(Om_dynamic * (1 + z_array)**3 + (1 - Om_dynamic))
    H_vacuum = H_dynamic * E_z_vac

    return H_vacuum, H_lcdm, H_shoes

# ==========================================
# 3. GENERATE & PLOT
# ==========================================
print("Simulating Vacuum Phase Transition (Dynamic Energy Release Model)...")
z_eval = np.linspace(0, 2.5, 500)
H_vac, H_lcdm, H_shoes = get_hubble_evolution(z_eval)

print(f"--- TRANSITION DIAGNOSTICS ---")
print(f"H0 (Planck Base):    {H_lcdm[0]:.2f} km/s/Mpc")
print(f"H0 (Vacuum Late):    {H_vac[0]:.2f} km/s/Mpc (Target: {H_OBS})")
print(f"H_fast (Vacuum Early Ceiling): {H_FAST} km/s/Mpc")
print(f"Transition z:        {Z_TRANS}")
print(f"Late-time Drag:      Om_eff = {Om_EFF}")

# Plotting
plt.figure(figsize=(10, 6))

plt.plot(z_eval, H_lcdm, 'k--', label='Planck 2018 ($H_0=67.4$)')
plt.plot(z_eval, H_shoes, 'g:', linewidth=2, label='SH0ES ($H_0=73.0$)')
plt.plot(z_eval, H_vac, 'r-', linewidth=3, label=f'Vacuum Model (Terminal $H_0={H_vac[0]:.1f}$)')

# Highlight the Transition Zone
# Added 'r' prefix to treat as raw string for LaTeX symbols
plt.axvspan(Z_TRANS - WIDTH, Z_TRANS + WIDTH, color='red', alpha=0.1, label=r'Phase Transition ($z \approx 0.65$)')
plt.axvline(Z_TRANS, color='red', linestyle=':', alpha=0.5)

plt.xlabel('Redshift $z$', fontsize=12)
plt.ylabel('$H(z)$ [km/s/Mpc]', fontsize=12)
plt.title(r'Vacuum Elastodynamics: Phase Transition Deceleration at $z \approx 0.65$', fontsize=14)
plt.legend(loc='upper left')
plt.xlim(0, 2.0)
plt.ylim(60, 200)
plt.grid(alpha=0.3)

plt.savefig('Figure2_Hubble_Transition_Updated.pdf')
plt.show()
