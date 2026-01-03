import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

print("--- ISW STABILITY TEST (Scientifically Corrected) ---")

# ==========================================
# 1. PARAMETERS (Add 46)
# ==========================================
Om0 = 0.315
Z_TRANS = 0.65
WIDTH = 0.10

# "SCIENTIFICALLY CORRECT" PARAMETER SELECTION
# 1. We remove the arbitrary 'S8_SCALING = 7.4'. It is unphysical.
# 2. We use ETA_LATE = 0.157 (Proton Load) as the physical ground state.
#    (Section 9.25 uses 0.17 as an effective average, but 0.157 is the fundamental value).

ETA_PHYSICAL = 0.157  # The Proton Load (Fundamental)
ETA_LIMIT    = 0.21   # The Yield Limit (Safety Ceiling)

# We use the physical value for the primary test
ETA_TEST = ETA_PHYSICAL 

# ==========================================
# 2. PHYSICS ENGINE
# ==========================================
def sigmoid(z):
    arg = (z - Z_TRANS) / WIDTH
    return np.where(arg > 50, 0.0, 1.0 / (1.0 + np.exp(arg)))

def system_ode(y, a, model='std'):
    delta, d_delta = y
    z = 1.0/a - 1.0
    E = np.sqrt(Om0 * a**-3 + (1 - Om0))

    # Standard Friction
    dE_da = -1.5 * Om0 * a**-4 / E
    hubble_friction = 3.0/a + dE_da/E

    # Vacuum Viscosity
    if model == 'vac':
        # Dynamic Profile: Viscosity turns ON at late times
        eta_eff = ETA_TEST * sigmoid(z)
        friction = hubble_friction + eta_eff / a
    else:
        friction = hubble_friction

    source = 1.5 * Om0 / (a**5 * E**2)
    return [d_delta, -friction * d_delta + source * delta]

# ==========================================
# 3. RUN SIMULATION
# ==========================================
a_grid = np.linspace(0.1, 1.0, 500)
y0 = [a_grid[0], 1.0]

sol_std = odeint(system_ode, y0, a_grid, args=('std',))
sol_vac = odeint(system_ode, y0, a_grid, args=('vac',))

# ==========================================
# 4. ANALYSIS
# ==========================================
phi_std = sol_std[:, 0] / a_grid
phi_vac = sol_vac[:, 0] / a_grid

# Normalize at transition (z=0.65) to see late-time divergence
idx_trans = (np.abs(1/a_grid - 1 - Z_TRANS)).argmin()
norm_std = phi_std / phi_std[idx_trans]
norm_vac = phi_vac / phi_vac[idx_trans]

# Decay Rates at z=0
decay_std = (norm_std[-1] - norm_std[-50]) / (a_grid[-1] - a_grid[-50])
decay_vac = (norm_vac[-1] - norm_vac[-50]) / (a_grid[-1] - a_grid[-50])

isw_boost = (decay_vac / decay_std)**2

print(f"Parameter Used: Eta = {ETA_TEST} (Proton Load)")
print(f"Scaling Factor: REMOVED (Set to 1.0)")
print(f"-"*40)
print(f"Standard Decay: {decay_std:.4f}")
print(f"Vacuum Decay:   {decay_vac:.4f}")
print(f"ISW Boost:      {isw_boost:.2f}x")

if isw_boost < 1.3:
    print("VERDICT: PASS.")
else:
    print("VERDICT: TENSION. Decay is too fast.")
    
# ==========================================
# 5. PLOT
# ==========================================
plt.figure(figsize=(8,6))
z_plot = 1.0/a_grid - 1.0
plt.plot(z_plot, norm_std, 'k--', label='Standard Potential')
plt.plot(z_plot, norm_vac, 'r-', label=f'Vacuum Potential (Eta={ETA_TEST})')

plt.xlim(0, 1.5)
plt.xlabel('Redshift z')
plt.ylabel('Gravitational Potential (Normalized)')
plt.title('Integrated Sachs-Wolfe (ISW) Stability Check')
plt.gca().invert_xaxis()
plt.legend()
plt.grid(alpha=0.3)
plt.show()
