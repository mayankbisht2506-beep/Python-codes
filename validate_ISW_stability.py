import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# ==========================================
# 1. PARAMETERS (Add 59.pdf)
# ==========================================
Om0 = 0.315
ETA_DRAG = 0.17
Z_TRANS = 0.65

# ==========================================
# 2. GROWTH & POTENTIAL SOLVER
# ==========================================
def system_ode(y, a, model='std'):
    # y = [delta, delta_prime]
    delta, d_delta = y
    z = 1.0/a - 1.0
    E = np.sqrt(Om0 * a**-3 + (1 - Om0))

    # Standard Friction
    friction = (3/a + (-1.5 * Om0 * a**-4 / E**2))

    # Add Viscosity for Vacuum Model
    if model == 'vac' and z < Z_TRANS:
        friction += ETA_DRAG / a

    # Poisson Source Term
    source = 1.5 * Om0 / (a**5 * E**2)

    return [d_delta, -friction * d_delta + source * delta]

# Solve Growth
a_grid = np.linspace(0.1, 1.0, 100)
sol_std = odeint(system_ode, [0.1, 1.0], a_grid, args=('std',))
sol_vac = odeint(system_ode, [0.1, 1.0], a_grid, args=('vac',))

# Calculate Potential Evolution Phi ~ G * rho * delta / k^2 ~ delta / a
phi_std = sol_std[:, 0] / a_grid
phi_vac = sol_vac[:, 0] / a_grid

# Normalize to 1 at transition for comparison
idx_trans = (np.abs(1/a_grid - 1 - Z_TRANS)).argmin()
phi_std /= phi_std[idx_trans]
phi_vac /= phi_vac[idx_trans]

# ==========================================
# 3. RESULTS & VERDICT
# ==========================================
# ISW is proportional to (d_phi / d_t)^2.
# We check if the slope changes drastically.
slope_std = np.gradient(phi_std, a_grid)[-1]
slope_vac = np.gradient(phi_vac, a_grid)[-1]
isw_ratio = (slope_vac / slope_std)**2

print(f"--- ISW STABILITY TEST ---")
print(f"Standard Potential Decay Slope: {slope_std:.4f}")
print(f"Vacuum Potential Decay Slope:   {slope_vac:.4f}")
print(f"Relative ISW Power Multiplier:  {isw_ratio:.2f}x")

if isw_ratio < 2.0:
    print("VERDICT: PASS (Large-scale CMB preserved)")
else:
    print("VERDICT: FAIL (Low-ell CMB power too high)")
