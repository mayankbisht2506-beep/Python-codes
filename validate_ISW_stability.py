# Uncomment the line below if running in Google Colab / Jupyter
# !pip install -q scipy numpy matplotlib

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

print("--- VED OBSERVATIONAL REALITY: CONVOLVED SUPERVOID ISW TEST ---")
print("Objective: Compute the Survey-Convolved A_ISW across the Supervoid Window")
print("Engine: Zero-Parameter Covariant Geometry (Width=0.084)")
print("-" * 75)

# ==========================================
# 1. PARAMETERS (Exact Topological Roots)
# ==========================================
OM_PRIMORDIAL = 0.3116  # Exact Bare Density (Gravity Source)
OM_EFFECTIVE  = 0.3639  # Exact Macroscopic Inertial Load (Kinematic Drag)
ZETA_FLOOR    = 0.15683 # Vacuum Saturation Viscosity (Frenkel limit)
ZETA_PEAK     = 0.3116  # Percolation Jamming Limit (p_c)
Z_TRANS       = 0.641   # Exact Topological Phase Transition Redshift
WIDTH         = 0.084   # EXACT: Derived geometric Jacobian width

# ==========================================
# 2. PHYSICS ENGINE (Eq. 126)
# ==========================================
def get_sigmoid(z):
    arg = np.clip((Z_TRANS - z) / WIDTH, -100.0, 100.0)
    return 1.0 / (1.0 + np.exp(-arg))

def get_viscosity(z):
    late_trigger = get_sigmoid(z)
    base_viscosity = ZETA_FLOOR * late_trigger
    spike_arg = np.clip(-0.5 * ((z - Z_TRANS) / WIDTH)**2, -100.0, 0.0)
    spike = (ZETA_PEAK - ZETA_FLOOR) * np.exp(spike_arg)
    return base_viscosity + spike

def growth_ode_rigorous(y, a, model='lcdm'):
    delta, delta_prime = y
    z = 1.0 / a - 1.0
    
    if model == 'viscous':
        S_z = get_sigmoid(z)
        om_z = OM_PRIMORDIAL + (OM_EFFECTIVE - OM_PRIMORDIAL) * S_z
    else:
        om_z = OM_PRIMORDIAL

    E = np.sqrt(om_z * (1+z)**3 + (1 - om_z))
    dE_da = -1.5 * om_z * (a**-4) / E
    source = 1.5 * OM_PRIMORDIAL / (a**5 * E**2)

    if model == 'viscous':
        zeta = get_viscosity(z)
        friction_term = (1.0/a) + (dE_da/E) + (2.0/a) * (1.0 + zeta)**2.0
        return [delta_prime, -friction_term * delta_prime + source * delta]

    hubble_friction = 3.0/a + dE_da/E
    return [delta_prime, -hubble_friction * delta_prime + source * delta]

# ==========================================
# 3. RUN SIMULATION & EXTRACT POTENTIALS
# ==========================================
print("Solving Linear Perturbation ODEs...")
a_grid = np.linspace(0.1, 1.0, 2000)
y0 = [a_grid[0], 1.0]

sol_lcdm = odeint(growth_ode_rigorous, y0, a_grid, args=('lcdm',))
sol_visc = odeint(growth_ode_rigorous, y0, a_grid, args=('viscous',))

# Gravitational Potential Phi(a) ~ Omega_m * delta / a
phi_lcdm = OM_PRIMORDIAL * sol_lcdm[:, 0] / a_grid
phi_visc = OM_PRIMORDIAL * sol_visc[:, 0] / a_grid

# Normalize at early matter-dominated regime (z=9) where Phi is constant
norm_lcdm = phi_lcdm / phi_lcdm[0]
norm_visc = phi_visc / phi_visc[0]

z_grid = 1.0 / a_grid - 1.0

from scipy.integrate import odeint, trapezoid  # Modern, non-deprecated import

# ==========================================
# 4. ASTROPHYSICAL OBSERVATIONAL CONVOLUTION
# ==========================================
# Flip arrays so redshift z is strictly ascending: 0.0 -> 9.0
z_asc = np.flip(z_grid)
norm_lcdm_asc = np.flip(norm_lcdm)
norm_visc_asc = np.flip(norm_visc)

# Physical potential decay rate per unit redshift: |dPhi / dz|
dphi_dz_lcdm = np.abs(np.gradient(norm_lcdm_asc, z_asc))
dphi_dz_visc = np.abs(np.gradient(norm_visc_asc, z_asc))

# Empirical Supervoid Stacking Survey Selection Window (Granett et al. / DES)
Z_VOID_MEAN = 0.52
Z_VOID_SIGMA = 0.18
W_survey = np.exp(-0.5 * ((z_asc - Z_VOID_MEAN) / Z_VOID_SIGMA) ** 2)
W_survey /= trapezoid(W_survey, z_asc)  # Strictly positive normalization

# Evaluate the True Line-of-Sight Integrated ISW Signal
signal_lcdm = trapezoid(W_survey * dphi_dz_lcdm, z_asc)
signal_visc = trapezoid(W_survey * dphi_dz_visc, z_asc)

A_ISW_convolved = signal_visc / signal_lcdm

# Asymptotic endpoints for comparison
mask_local = z_asc <= 0.10
A_ISW_local = (norm_visc_asc[mask_local][0] - norm_visc_asc[mask_local][-1]) / (
    norm_lcdm_asc[mask_local][0] - norm_lcdm_asc[mask_local][-1]
)

idx_trans = np.argmin(np.abs(z_asc - Z_TRANS))
A_ISW_transition_peak = (norm_visc_asc[idx_trans] - norm_visc_asc[0]) / (
    norm_lcdm_asc[idx_trans] - norm_lcdm_asc[0]
)

# ==========================================
# 5. RESULTS & VERDICT
# ==========================================
print("\n" + "=" * 65)
print("             ISW MULTI-SCALE RHEOLOGICAL AUDIT")
print("=" * 65)
print(f"1. Terminal Local Boundary (z <= 0.10):        {A_ISW_local:.3f}x")
print(f"2. Transition Bound Cumulative Drop (z <= 0.64):{A_ISW_transition_peak:.3f}x")
print("-" * 65)
print(
    f"3. TRUE OBSERVATIONAL REALITY (Convolved):      {A_ISW_convolved:.3f}x  <-- [TARGET]"
)
print("=" * 65)

# ==========================================
# 6. GENERATE PUBLICATION PLOT
# ==========================================
fig, ax1 = plt.subplots(figsize=(10, 6))

ax1.set_xlabel("Redshift $z$", fontsize=12)
ax1.set_ylabel(
    r"Physical Potential Decay Rate $|d\Phi/dz|$", fontsize=12, color="black"
)
ax1.plot(
    z_asc,
    dphi_dz_lcdm,
    "k--",
    linewidth=2,
    label=r"Standard $\Lambda$CDM ($|d\Phi/dz|$)",
)
ax1.plot(
    z_asc,
    dphi_dz_visc,
    "r-",
    linewidth=2.5,
    label=r"Vacuum Elastodynamics ($|d\Phi/dz|$)",
)
ax1.set_xlim(1.2, 0.0)  # Look back in time: 1.2 -> 0.0
ax1.set_ylim(0, np.max(dphi_dz_visc) * 1.2)
ax1.axvline(
    Z_TRANS, color="grey", linestyle=":", label=rf"Phase Transition ($z={Z_TRANS}$)"
)
ax1.grid(True, alpha=0.3)

# Secondary axis showing the Survey Window Function
ax2 = ax1.twinx()
color = "tab:blue"
ax2.set_ylabel(
    r"Survey Selection Window $W(z)$ [Granett et al.]",
    fontsize=12,
    color=color,
)
ax2.fill_between(
    z_asc, W_survey, color=color, alpha=0.15, label="Supervoid Catalog Window"
)
ax2.tick_params(axis="y", labelcolor=color)
ax2.set_ylim(0, np.max(W_survey) * 1.3)  # Now strictly positive and clean!

# Combined Legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=10)

plt.title(
    rf"Observational ISW Resolution: Convolved $A_{{ISW}} = {A_ISW_convolved:.2f}$",
    fontsize=14,
)
plt.tight_layout()
plt.savefig("Figure_ISW_Convolved_Observational.pdf", dpi=300)
print("\nPlot cleanly saved to 'Figure_ISW_Convolved_Observational.pdf'")
plt.show()
