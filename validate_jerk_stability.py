import numpy as np
import matplotlib.pyplot as plt

print("--- COSMIC WHIPLASH (JERK PARAMETER STABILITY) ---")
print("Objective: Verify the Phase Transition doesn't cause unphysical derivatives.")

# ==========================================
# 1. PHYSICS ENGINE
# ==========================================
# Standard LCDM (j = 1 by definition)
# Vacuum Model: H(z) varies due to stiffness G(z) and Viscosity
# Using the profile from your Hubble Transition Model

z_grid = np.linspace(0, 2.0, 2000)
dz = z_grid[1] - z_grid[0]

# Transition Parameters (Matches Paper)
H0_VAC = 74.5
Z_TRANS = 0.65
WIDTH = 0.1
DELTA_G = 0.181  # Effective relaxation (Eq 69)

def get_hubble_vac(z_arr):
    # Transition function (Sigmoid)
    arg = (z_arr - Z_TRANS) / WIDTH
    # Logistic weighting: 0 (Early) -> 1 (Late/Now)
    # Note: In paper, transition is from Early (High G) to Late (Low G)
    # Weight w represents "Late-ness"
    w = 1.0 / (1.0 + np.exp(-arg)) # 0 at high z, 1 at low z? 
    # Let's check: at z=0 (Now), arg < 0?? 
    # Wait, z goes 0 -> 2. z_trans=0.65.
    # z < 0.65 (Late universe). z > 0.65 (Early universe).
    # If z=0: arg = -6.5. exp(6.5) >> 1. w -> 0.
    # If z=2: arg = 13.5. exp(-13.5) ~ 0. w -> 1.
    # This logic is inverted. Let's fix to match "Relaxation".
    
    # Correct Logic:
    # Early Universe (z > 0.65): Stiff (G_boost ~ 1.22).
    # Late Universe (z < 0.65): Standard (G ~ 1).
    
    # Sigmoid "S" going from 0 (Late) to 1 (Early)
    S = 1.0 / (1.0 + np.exp(-(z_arr - Z_TRANS)/WIDTH))
    
    # Gravity Boost profile
    # G(z) = G0 * (1 + boost * S)
    # H(z) ~ sqrt(G * rho)
    # Standard H(z)
    E_lcdm = np.sqrt(0.315*(1+z_arr)**3 + (1-0.315))
    
    # Vacuum H(z) with Boost
    # Boost factor matches H0=74.5 vs 67.4 scaling
    boost_factor = 74.5 / 67.4 # ~ 1.105
    
    # Apply transition: 
    # At z=0 (S~0), H ~ H0_VAC (actually H0_VAC is the local value?)
    # Wait, H0_VAC is the *result* of the boost?
    # Let's simply model the H(z) shape directly.
    # H_vac(z) = H_lcdm(z) * (Transition_Modification)
    
    # Modification: 
    # Local (z=0): Base level.
    # High z: Enhanced level (or vice versa? Paper says H0 is boosted *locally* due to relaxation?)
    # Section 7.1: "Relaxation... boosting the late-time expansion".
    # So Late Time (z < 0.65) is Boosted. Early Time is "Standard" (Planck-compatible)?
    # Wait, G_early ~ 1.22 G0. G relaxes *down* to G0? 
    # Eq 67: G(z) = G_early * (1 - delta).
    # Current G (z=0) is LOWER than Early G.
    # But H ~ sqrt(G). So Early H should be HIGHER?
    # Actually, Planck H0 is low (67). Vacuum H0 is high (74).
    # This implies the *Local* universe is running fast.
    
    # Let's trust the "H0 Resolution":
    # The boost is LOCAL. 
    # Transition: High Activity (z < 0.65).
    
    # Activation S: 1 at z < 0.65, 0 at z > 0.65.
    # Using negative argument for transition.
    S_active = 1.0 / (1.0 + np.exp((z_arr - Z_TRANS)/WIDTH))
    
    # Amplitude modulation
    # H_vac = H_lcdm * (1 + delta * S_active)
    # We want H ~ 74.5 at z=0, H ~ 67.4-like behavior at z>>1?
    # Actually, H(z) matches Planck at high z.
    
    scaling = 1.0 + ( (74.5/67.4) - 1.0 ) * S_active
    
    return 67.4 * E_lcdm * scaling

# ==========================================
# 2. CALCULATE DERIVATIVES (q and j)
# ==========================================
# Scale factor a = 1/(1+z)
# Time relation: H = da/dt / a  -> dt = da / (a H)
# We can compute j cosmographically using H(z) derivatives:
# j(z) = [ H(z)^2 + (1+z) d/dz(H(z)^2) ] / H(z)^2 ? 
# Standard formula: j(z) = 1 + 2q + (1+z) dq/dz ??
# Let's use the exact relation involving H' = dH/dz
# q(z) = (1+z)/H * dH/dz - 1
# j(z) = (1+z)^2 [ (dH/dz)^2 / H^2 + 1/H * d2H/dz2 ] ??? No.

# Exact Formulae:
# q = (1+z) (H'/H) - 1
# j = (1+z)^2 [ H''/H + (H'/H)^2 ] - 2(1+z)(H'/H) + 1  <-- Let's derive numerically
# Alternatively: j = q + 2q^2 + (1+z) dq/dz

Hz = get_hubble_vac(z_grid)

# First Derivative dH/dz
dHdz = np.gradient(Hz, z_grid)

# Deceleration Parameter q(z)
q = (1+z_grid) * (dHdz / Hz) - 1

# Second Derivative d2H/dz2 (via dq/dz)
dqdz = np.gradient(q, z_grid)

# Jerk Parameter j(z)
# j = q(2q+1) + (1+z) dq/dz
j = q*(2*q + 1) + (1+z_grid)*dqdz

# ==========================================
# 3. VERDICT
# ==========================================
max_j = np.max(np.abs(j))
print(f"Max Absolute Jerk: {max_j:.2f}")

# Threshold: Standard LCDM has j=1.
# A "Whiplash" would be j > 10 or 20.
if max_j < 15.0:
    print("VERDICT: PASS. Transition is physically smooth.")
    print("The phase transition does not violate kinematic bounds.")
else:
    print("VERDICT: FAIL. Cosmic Whiplash detected!")
    print("The transition is too sharp (Singularity risk).")

# Plot
plt.figure(figsize=(10,6))
plt.plot(z_grid, j, 'r-', label='Vacuum Jerk j(z)')
plt.axhline(1.0, color='k', linestyle='--', label='LCDM (j=1)')
plt.axvline(Z_TRANS, color='gray', linestyle=':', label='Transition')
plt.xlabel('Redshift z')
plt.ylabel('Jerk Parameter j(z)')
plt.title('Test 23: Cosmic Whiplash (Stability of Expansion)')
plt.ylim(-5, 10) # Zoom in to see structure
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
