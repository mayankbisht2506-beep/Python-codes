import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. PARAMETERS (Quadruple Concordance)
# ==========================================
# Observational Constraints
H0_EARLY = 67.4     # Early Universe (Planck 2018)
H0_LATE  = 73.74    # Late Universe (Updated to match Abstract/Section 7.1) [cite: 8, 719]

# Error bands
ERR_PLANCK = 0.5
ERR_SHOES  = 1.02   # Updated from 1.0 to match text (±1.02) [cite: 39]

# Theoretical Parameters (Vacuum Elastodynamics)
Z_TRANSITION = 0.65     # Percolation Threshold (derived geometrically) [cite: 7, 37]
TRANSITION_WIDTH = 0.1  # Width of the phase transition

# ==========================================
# 2. MODEL FUNCTION
# ==========================================
def effective_h0(z):
    """
    Sigmoidal hardening of the vacuum (Cosmic Metallurgy).
        
    Physics:
        - High z (Early): Vacuum is 'Soft' (Fluid), G is High (~1.24 G0). 
          (Note: We plot the *effective* H0 inferred from this geometry)
                
        - Low z (Late): Vacuum 'Stiffens' (Crystal), G relaxes to G0.
    """
    # Sigmoid function: 1 at low z (Late), 0 at high z (Early)
    # The logic here is inverted in z space (low z = high completion)
    # We want sigmoid=1 when z is small (Late universe)
    # We want sigmoid=0 when z is large (Early universe)
    
    # Correct Sigmoid for Redshift:
    # 1 / (1 + exp((z - zt)/w))
    # If z = 0 (Late): exp(-6.5) ~ 0 -> 1/(1) = 1. (Correct)
    # If z = 2 (Early): exp(13.5) ~ inf -> 1/inf = 0. (Correct)
    sigmoid = 1.0 / (1.0 + np.exp((z - Z_TRANSITION) / TRANSITION_WIDTH))
    
    # Interpolate:
    # We observe H0=73.74 today (Stiff). The Early Universe 'looks' like H0=67.4.
    h0_eff = H0_EARLY + (H0_LATE - H0_EARLY) * sigmoid
    return h0_eff

# Generate Data
z_values = np.linspace(0, 2.0, 500)
h0_values = effective_h0(z_values)

# ==========================================
# 3. PLOTTING
# ==========================================
plt.figure(figsize=(10, 6))

# A. Error Bands
# Late Universe (SHOES)
plt.fill_between(z_values, H0_LATE - ERR_SHOES, H0_LATE + ERR_SHOES,
                 color='red', alpha=0.15, label=f'SH0ES (Late Universe): {H0_LATE} $\pm$ {ERR_SHOES}')
plt.axhline(H0_LATE, color='red', linestyle='--', alpha=0.5, linewidth=1)

# Early Universe (Planck)
plt.fill_between(z_values, H0_EARLY - ERR_PLANCK, H0_EARLY + ERR_PLANCK,
                 color='blue', alpha=0.15, label=f'Planck (Early Universe): {H0_EARLY} $\pm$ {ERR_PLANCK}')
plt.axhline(H0_EARLY, color='blue', linestyle='--', alpha=0.5, linewidth=1)

# B. Theory Curve
plt.plot(z_values, h0_values, 'k-', linewidth=3, label='Vacuum Elastodynamics Prediction')

# C. Annotations (CORRECTED for Hardening Narrative)
plt.arrow(Z_TRANSITION, 71.5, 0, -2.5, head_width=0.05, head_length=0.5, fc='k', ec='k')
plt.text(Z_TRANSITION + 0.05, 70.0, f'Vacuum Hardening\nPhase Transition ($z \\approx {Z_TRANSITION}$)', fontsize=10)

# FIX: Late Universe is STIFF (Low G)
plt.text(0.05, 74.2, 'Local "Stiff" Vacuum\n(Crystalline, Low G)', fontsize=10, color='darkred', fontweight='bold')

# FIX: Early Universe is SOFT (High G)
plt.text(1.3, 66.0, 'Primordial "Soft" Vacuum\n(Fluid, High G)', fontsize=10, color='darkblue', fontweight='bold')

# D. Formatting
plt.xlim(0, 2.0)
plt.ylim(65, 76) # Adjusted limits to fit 73.74 comfortably
plt.xlabel('Redshift ($z$)', fontsize=12)
plt.ylabel('Effective Hubble Constant $H_0$ (km/s/Mpc)', fontsize=12)
plt.title('Resolution of the Hubble Tension via Cosmic Metallurgy', fontsize=14)
plt.legend(loc='center right')
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Save
plt.savefig('Figure2_Corrected_Hardening.png', dpi=300)
plt.show()
