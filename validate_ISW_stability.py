import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# ==========================================
# 1. PARAMETERS (Unified Add 33)
# ==========================================
Om0 = 0.315
Z_TRANS = 0.65
WIDTH = 0.15

# LEPTON SUM RULE UPDATES
ETA_MICRO = 0.21         # Fixed by Leptons
S8_SCALING = 7.4         # Fixed by Weak Lensing (S8)

# The Effective Macroscopic Drag
# This is the actual friction felt by the potentials
ETA_MACRO = ETA_MICRO * S8_SCALING  # approx 1.55

# ==========================================
# 2. GROWTH & POTENTIAL SOLVER
# ==========================================
def sigmoid(z):
    arg = (z - Z_TRANS) / WIDTH
    return np.where(arg > 50, 0.0, 1.0 / (1.0 + np.exp(arg)))

def system_ode(y, a, model='std'):
    # y = [delta, delta_prime]
    delta, d_delta = y
    z = 1.0/a - 1.0
    E = np.sqrt(Om0 * a**-3 + (1 - Om0))

    # Standard Friction (Hubble Drag)
    dE_da = -1.5 * Om0 * a**-4 / E
    hubble_friction = 3.0/a + dE_da/E

    # Vacuum Viscosity (The S8 Solution)
    if model == 'vac':
        # Dynamic Profile
        eta_eff = ETA_MACRO * sigmoid(z)
        # Add linear drag term
        friction = hubble_friction + eta_eff / a
    else:
        friction = hubble_friction

    # Poisson Source Term
    source = 1.5 * Om0 / (a**5 * E**2)

    return [d_delta, -friction * d_delta + source * delta]

# Solve Growth
a_grid = np.linspace(0.1, 1.0, 500)
# Initial Conditions: delta ~ a at early times
y0 = [a_grid[0], 1.0]

sol_std = odeint(system_ode, y0, a_grid, args=('std',))
sol_vac = odeint(system_ode, y0, a_grid, args=('vac',))

# ==========================================
# 3. ISW ANALYSIS
# ==========================================
# Gravitational Potential Phi ~ (Delta / a)
# (Ignoring constants G, rho, k^2 as they cancel in ratio)
phi_std = sol_std[:, 0] / a_grid
phi_vac = sol_vac[:, 0] / a_grid

# Normalize potentials to match at transition (z=0.65)
# We want to see how they diverge LATE in time
idx_trans = (np.abs(1/a_grid - 1 - Z_TRANS)).argmin()
norm_std = phi_std / phi_std[idx_trans]
norm_vac = phi_vac / phi_vac[idx_trans]

# Calculate Decay Rate (dPhi/dt proxies)
# ISW Power is proportional to the square of the time derivative of Phi.
# A faster decay means more ISW power.
decay_std = np.gradient(norm_std, a_grid)[-1]
decay_vac = np.gradient(norm_vac, a_grid)[-1]

# ISW Power Ratio (approximate)
isw_boost = (decay_vac / decay_std)**2

print(f"--- ISW STABILITY TEST (Corrected) ---")
print(f"Input Parameters: Eta={ETA_MICRO}, Scaling={S8_SCALING}")
print(f"Standard Potential Decay Slope: {decay_std:.4f}")
print(f"Vacuum Potential Decay Slope:   {decay_vac:.4f}")
print(f"ISW Power Multiplier:           {isw_boost:.2f}x")

# ==========================================
# 4. VERDICT & PLOT
# ==========================================
# ISW data is noisy (cosmic variance). 
# Generally, a boost < 3.0x is acceptable at lowest multipoles.
if isw_boost < 3.0:
    print("VERDICT: PASS. The ISW enhancement is within cosmic variance.")
else:
    print("VERDICT: TENSION. Potential decays too fast.")

plt.figure(figsize=(8,6))
z_plot = 1.0/a_grid - 1.0
plt.plot(z_plot, norm_std, 'k--', label=r'Standard Potential $\Phi$')
plt.plot(z_plot, norm_vac, 'r-', linewidth=2, label=r'Vacuum Potential $\Phi$')



plt.xlim(0, 1.5)
plt.xlabel('Redshift z')
plt.ylabel('Gravitational Potential (Normalized)')
plt.title('Integrated Sachs-Wolfe (ISW) Effect Check')
plt.gca().invert_xaxis()
plt.legend()
plt.grid(alpha=0.3)
plt.show()
