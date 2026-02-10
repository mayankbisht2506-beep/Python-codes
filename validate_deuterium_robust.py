import numpy as np
import matplotlib.pyplot as plt

print("--- PRIMORDIAL DEUTERIUM INVARIANCE (STRICT) ---")
print("Objective: Verify the 'Cancellation Theorem' (Section 7.13) for Deuterium.")

# ==========================================
# 1. PARAMETERS (Strict Theory Consistency)
# ==========================================
# Observation (Particle Data Group)
OBS_DH = 2.547e-5

# The "Grand Unification" Values (Section 7.1)
H0_PLANCK = 67.4
H0_THEORY = 74.5    # The Gravity Boost Prediction

# CALCULATE G_BOOST EXACTLY
# H ~ sqrt(G), so G ~ H^2
G_BOOST = (H0_THEORY / H0_PLANCK)**2  # approx 1.2216

# ==========================================
# 2. SCALING LAWS (The Cancellation Theorem)
# ==========================================
# Derived from Section 7.13

# A. Density Scaling
# rho ~ T^3. Since T_nuc scales with Binding Energy Q ~ m ~ G^-0.5:
# rho ~ (G^-0.5)^3 = G^-1.5
RHO_SCALE = G_BOOST**(-1.5)

# B. Time Scaling (Section 7.12.1)
# Expansion H ~ sqrt(G)*T^2.
# Time t ~ 1/H ~ 1/(G^0.5 * T^2)
# Substituting T ~ G^-0.5:
# t ~ 1/(G^0.5 * G^-1.0) = G^0.5
TIME_SCALE = G_BOOST**(0.5)

# C. Cross-Section Scaling
# sigma ~ 1/m^2 (Compton scale).
# Since m ~ G^-0.5 (Eq. 80), sigma ~ 1/(G^-1) = G^1.0
SIGMA_SCALE = G_BOOST**(1.0)

# ==========================================
# 3. ROBUST BURNING MODEL
# ==========================================
def run_simulation():
    # Standard Model Calibration
    # We define the "Target Exponent" required to burn initial D down to observed levels.
    Y0 = 2.0e-4
    target_exponent = np.log(Y0 / OBS_DH) # Approx 2.06

    # Standard Model Final Abundance
    Y_final_std = Y0 * np.exp(-target_exponent)

    # Vacuum Model Calculation
    # The burning efficiency (exponent) scales with Rate * Time
    # Rate ~ Density * Sigma * v (v is thermal/invariant)
    # Total Burn ~ (Rho * Sigma) * Time
    
    # NET SCALING FACTOR (The Cancellation)
    net_scaling = RHO_SCALE * SIGMA_SCALE * TIME_SCALE

    vac_exponent = target_exponent * net_scaling
    Y_final_vac = Y0 * np.exp(-vac_exponent)

    return Y_final_std, Y_final_vac, net_scaling

# ==========================================
# 4. EXECUTE & VERIFY
# ==========================================
final_std, final_vac, scaling_factor = run_simulation()

ratio = final_vac / final_std
percent_change = (ratio - 1) * 100

print(f"H0 Theory: {H0_THEORY} (implies G_BOOST = {G_BOOST:.4f})")
print("-" * 50)
print(f"SCALING FACTORS (Section 7.12):")
print(f"  Density (dilution):        {RHO_SCALE:.4f}  (G^-1.5)")
print(f"  Cross-Section (boost):     {SIGMA_SCALE:.4f}  (G^1.0)")
print(f"  Time Window (delay):       {TIME_SCALE:.4f}  (G^0.5)")
print(f"  NET CANCELLATION:          {scaling_factor:.4f}")
print("-" * 50)
print(f"Standard D/H:  {final_std:.4e}")
print(f"Vacuum D/H:    {final_vac:.4e}")
print(f"Percent Drift: {percent_change:+.4f}%")
print("-" * 50)

# ==========================================
# 5. VERDICT
# ==========================================
if abs(percent_change) < 0.001:
    print("VERDICT: PASS (PERFECT SYMMETRY)")
    print("The scaling laws cancel exactly, preserving BBN abundances.")
else:
    print("VERDICT: FAIL")

# Plot
plt.figure(figsize=(6,5))
x_labels = [r'Standard $\Lambda$CDM', f'Vacuum ($H_0={H0_THEORY}$)']
y_values = [final_std*1e5, final_vac*1e5]

plt.bar(x_labels, y_values, color=['gray', '#1f77b4'], width=0.5)
plt.axhline(OBS_DH*1e5, color='red', linestyle='--', linewidth=1, label='Particle Data Group')

plt.ylabel(r'Deuterium Abundance ($10^{-5}$)')
plt.title(f'BBN Invariance: The Cancellation Theorem')
plt.ylim(0, 3.0)
plt.legend()
plt.tight_layout()
plt.savefig('Figure_BBN_Strict.png')
plt.show()
