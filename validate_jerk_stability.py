import numpy as np
import matplotlib.pyplot as plt

print("--- COSMIC WHIPLASH (JERK PARAMETER STABILITY) ---")
print("Objective: Verify the Phase Transition doesn't cause unphysical derivatives.")

# ==========================================
# 1. PHYSICS ENGINE
# ==========================================
# Grid Setup
z_grid = np.linspace(0, 2.0, 2000)

# Transition Parameters (Matches Paper)
H0_VAC = 74.5
Z_TRANS = 0.65
WIDTH = 0.1

def get_hubble_vac(z_arr):
    """
    Models the Vacuum Phase Transition.
    Late Universe (z < 0.65): Lattice relaxes, boosting H(z).
    Early Universe (z > 0.65): Lattice is stiff, matches Planck LCDM.
    """
    # Standard LCDM Baseline (Planck 2018 parameters)
    E_lcdm = np.sqrt(0.315*(1+z_arr)**3 + (1-0.315))
    
    # Transition Function (Sigmoid)
    # S = 1 at z=0 (Late Time / Boosted)
    # S = 0 at z>>0.65 (Early Time / Standard)
    S_active = 1.0 / (1.0 + np.exp((z_arr - Z_TRANS)/WIDTH))
    
    # Apply Boost
    # Scales H0 from 67.4 (Planck) to 74.5 (Vacuum) smoothly
    scaling = 1.0 + ( (74.5/67.4) - 1.0 ) * S_active
    
    return 67.4 * E_lcdm * scaling

# ==========================================
# 2. CALCULATE DERIVATIVES (q and j)
# ==========================================
# H(z) Array
Hz = get_hubble_vac(z_grid)

# First Derivative dH/dz
dHdz = np.gradient(Hz, z_grid)

# Deceleration Parameter q(z)
# Formula: q = (1+z)/H * (dH/dz) - 1
q = (1+z_grid) * (dHdz / Hz) - 1

# Derivative of q (dq/dz)
dqdz = np.gradient(q, z_grid)

# Jerk Parameter j(z)
# Formula: j = q(2q+1) + (1+z)*(dq/dz)
j = q*(2*q + 1) + (1+z_grid)*dqdz

# ==========================================
# 3. VERDICT
# ==========================================
max_j = np.max(np.abs(j))
print(f"Max Absolute Jerk: {max_j:.2f}")

# Threshold Analysis
if max_j < 15.0:
    print("VERDICT: PASS. Transition is physically smooth.")
else:
    print("VERDICT: FAIL. Cosmic Whiplash detected (Singularity risk).")

# Plot
plt.figure(figsize=(10,6))
plt.plot(z_grid, j, 'r-', linewidth=2, label='Vacuum Jerk j(z)')
plt.axhline(1.0, color='k', linestyle='--', label='LCDM Baseline (j=1)')
plt.axvline(Z_TRANS, color='gray', linestyle=':', label='Phase Transition (z=0.65)')
plt.xlabel('Redshift z')
plt.ylabel('Jerk Parameter j(z)')
plt.title(f'Stability Analysis: Jerk Parameter Evolution (Width={WIDTH})')
plt.ylim(-5, 10) 
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
