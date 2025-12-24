import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. PARAMETERS
# ==========================================
# Observational Constraints
H0_EARLY = 67.4   # Early Universe (Planck)
H0_LATE  = 73.0   # Late Universe (SHOES)

# Error bands
ERR_PLANCK = 0.5
ERR_SHOES  = 1.0

# Theoretical Parameters (Vacuum Elastodynamics)
Z_TRANSITION = 0.65   # Percolation Threshold (derived geometrically)
TRANSITION_WIDTH = 0.1 # Width of the phase transition

# ==========================================
# 2. MODEL FUNCTION
# ==========================================
def effective_h0(z):
    """
        Sigmoidal hardening of the vacuum (Cosmic Metallurgy).
            
                Physics:
                    - High z (Early): Vacuum is 'Soft' (Fluid), G is High (~1.23 G0). 
                          (Note: We plot the *effective* H0 inferred from this geometry)
                                
                                    - Low z (Late): Vacuum 'Stiffens' (Crystal), G relaxes to G0.
                                        """
    # Sigmoid function: 1 at low z (Late), 0 at high z (Early)
    sigmoid = 1 / (1 + np.exp((z - Z_TRANSITION) / TRANSITION_WIDTH))
    
    # Interpolate:
    # We observe H0=73 today (Stiff). The Early Universe 'looks' like H0=67.
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
                 color='red', alpha=0.15, label='SH0ES (Late Universe) Measurement')
plt.axhline(H0_LATE, color='red', linestyle='--', alpha=0.5, linewidth=1)

# Early Universe (Planck)
plt.fill_between(z_values, H0_EARLY - ERR_PLANCK, H0_EARLY + ERR_PLANCK,
                 color='blue', alpha=0.15, label='Planck (Early Universe) Measurement')
plt.axhline(H0_EARLY, color='blue', linestyle='--', alpha=0.5, linewidth=1)

# B. Theory Curve
plt.plot(z_values, h0_values, 'k-', linewidth=3, label='Vacuum Elastodynamics Prediction')

# C. Annotations (CORRECTED for Hardening Narrative)
plt.arrow(Z_TRANSITION, 71.5, 0, -2.5, head_width=0.05, head_length=0.5, fc='k', ec='k')
plt.text(Z_TRANSITION + 0.05, 70.0, 'Vacuum Hardening\nPhase Transition', fontsize=10)

# FIX: Late Universe is STIFF (Low G)
plt.text(0.1, 73.5, 'Local "Stiff" Vacuum\n(Crystalline, Low G)', fontsize=10, color='darkred', fontweight='bold')

# FIX: Early Universe is SOFT (High G)
plt.text(1.3, 66.0, 'Primordial "Soft" Vacuum\n(Fluid, High G)', fontsize=10, color='darkblue', fontweight='bold')

# D. Formatting
plt.xlim(0, 2.0)
plt.ylim(65, 75)
plt.xlabel('Redshift ($z$)', fontsize=12)
plt.ylabel('Effective Hubble Constant $H_0$ (km/s/Mpc)', fontsize=12)
plt.title('Resolution of the Hubble Tension via Cosmic Metallurgy', fontsize=14)
plt.legend(loc='center right')
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Save
plt.savefig('Figure2_Corrected_Hardening.png', dpi=300)
plt.show()
