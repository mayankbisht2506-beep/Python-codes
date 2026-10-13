# Uncomment the line below if running in Google Colab / Jupyter
# !pip install numpy matplotlib

import numpy as np
import matplotlib.pyplot as plt

print("--- SCALAR-TENSOR SCREENING AUDIT ---")
print("Objective: Verify Cassini compliance via Vainshtein Kinetic Screening (Sec 8.1.1)")
print("Engine: Non-linear Derivative Interactions (Galileon/Vainshtein Profile)")
print("-" * 70)

# ==========================================
# 1. PHYSICAL CONSTANTS
# ==========================================
# Masses & Distances
M_SUN = 1.989e30            # Solar mass (kg)
M_PLANCK = 2.176e-8         # Planck mass (kg)
AU_METERS = 1.496e11        # 1 Astronomical Unit (meters)
PC_METERS = 3.086e16        # 1 Parsec (meters)

# Observational Limits
CASSINI_LIMIT = 2.0e-5      # Upper bound on post-Newtonian deviation |gamma - 1|

# VED Interaction Scale (Derived from macroscopic yielding transition)
# Lambda_IR ~ (M_p * H_0^2)^(1/3) ~ 10^-13 eV
# 1 eV^-1 = 1.97e-7 meters -> 10^-13 eV^-1 = 1.97e6 meters
LAMBDA_IR_INV = 1.97e6      # Inverse interaction scale (meters)

# ==========================================
# 2. VAINSHTEIN SCREENING ENGINE
# ==========================================
def get_vainshtein_radius(M):
    """
    Calculates the Vainshtein radius for a central mass M.
    r_V = (M / M_P)^(1/3) * Lambda_{IR}^{-1}
    """
    return (M / M_PLANCK)**(1.0/3.0) * LAMBDA_IR_INV

def get_shapiro_deviation(r, r_v, n=2):
    """
    Calculates the fractional deviation in the post-Newtonian parameter gamma.
    Inside the Vainshtein sphere (r << r_v), deviation scales as (r / r_V)^n.
    We assume the minimal quadratic scaling (n=2) typical of cubic Galileons.
    """
    return (r / r_v)**n

# ==========================================
# 3. EXECUTION
# ==========================================
# Calculate Sun's Vainshtein Sphere
r_v_sun = get_vainshtein_radius(M_SUN)
r_v_sun_pc = r_v_sun / PC_METERS

# Evaluate at Cassini orbit (1 AU)
ratio_cassini = AU_METERS / r_v_sun
deviation_cassini = get_shapiro_deviation(AU_METERS, r_v_sun)

# ==========================================
# 4. RESULTS & AUDIT
# ==========================================
print(f"Central Source:       1 Solar Mass (M_sun)")
print(f"Vainshtein Radius:    {r_v_sun:.2e} meters")
print(f"                      (~{r_v_sun_pc:.0f} pc)")
print("-" * 70)
print(f"Target Distance:      1 AU ({AU_METERS:.2e} meters)")
print(f"Radial Ratio (r/r_V): {ratio_cassini:.2e}")
print(f"Calculated Deviation: {deviation_cassini:.2e}")
print(f"Cassini Upper Limit:  {CASSINI_LIMIT:.2e}")
print("-" * 70)

if deviation_cassini < CASSINI_LIMIT:
    print("VERDICT: PASS (CASSINI SCREENING CONFIRMED)")
    print("The macroscopic hyperuniform stiffness natively generates a massive")
    print("Vainshtein shielding sphere (~287 pc) around the Sun. The scalar fifth")
    print("force is kinetically frozen locally, passing Cassini limits by 11 orders")
    print("of magnitude without requiring density-dependent parameter tuning.")
else:
    print("VERDICT: FAIL")

# ==========================================
# 5. VISUALIZATION
# ==========================================
# Create radial grid from 0.01 AU to 10,000 pc
r_grid = np.logspace(np.log10(AU_METERS * 0.01), np.log10(PC_METERS * 10000), 500)
deviations = get_shapiro_deviation(r_grid, r_v_sun)

plt.figure(figsize=(9, 6))

# Plot the deviation curve
plt.loglog(r_grid, deviations, 'purple', linewidth=2.5, label=r'VED Vainshtein Profile $\gamma - 1 \approx (r/r_V)^2$')

# Shaded region for unscreened domain (r > r_V)
# In reality, it asymptotes to O(1), we cap the plot logic to highlight the transition
plt.fill_between(r_grid, deviations, 1.0, where=(r_grid > r_v_sun), color='gray', alpha=0.2, label='Unscreened Regime')

# Vertical markers
plt.axvline(AU_METERS, color='orange', linestyle='--', label=f'Cassini Orbit (1 AU)')
plt.axvline(r_v_sun, color='blue', linestyle='-.', label=f'Vainshtein Radius (~{r_v_sun_pc:.0f} pc)')

# Horizontal constraint
plt.axhline(CASSINI_LIMIT, color='red', linestyle='-', label=f'Cassini Constraint ($2 \\times 10^{{-5}}$)')
plt.fill_between(r_grid, CASSINI_LIMIT, 1e2, color='red', alpha=0.1)

# Formatting
plt.ylim(1e-20, 1e2)
plt.xlim(r_grid[0], r_grid[-1])
plt.xlabel('Distance from Sun $r$ [meters]', fontsize=12)
plt.ylabel(r'Post-Newtonian Deviation $|\gamma - 1|$', fontsize=12)
plt.title('Vainshtein Kinetic Screening of the VED Scalar Field', fontsize=14)
plt.legend(loc='upper left', fontsize=10)
plt.grid(True, which="both", ls=":", alpha=0.5)

plt.tight_layout()
plt.savefig('Figure_Vainshtein_Screening.pdf', dpi=300)
print("\nPlot saved as 'Figure_Vainshtein_Screening.pdf'")
plt.show()
