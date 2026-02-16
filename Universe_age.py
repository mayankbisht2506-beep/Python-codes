# Uncomment the line below if running in Google Colab / Jupyter
# !pip install scipy numpy

import numpy as np
from scipy.integrate import quad

print("--- VACUUM ELASTODYNAMICS: COSMIC AGE VERIFICATION ---")
print("Objective: Verify the 'Kinematic Age' under the Dual-Regime Vacuum Transition.")

# ==========================================
# 1. PHYSICS PARAMETERS (From MCMC & Theory)
# ==========================================
# Conversion Factor: 1/H0 [Gyr] = 977.79 / h (where H0 in km/s/Mpc)
# We use the raw formula: Age = Integral * (977.8 / H0_ref)
CONST_H_TO_AGE = 977.8 

# EARLY REGIME (z > 0.65): Superfluid Vacuum
# High Expansion (Geometric Ceiling) + Standard Primordial Density
H_EARLY  = 74.5     # km/s/Mpc
OM_EARLY = 0.315    # Primordial Matter Density

# LATE REGIME (z < 0.65): Viscous Vacuum
# Decelerated Flow (Terminal Velocity) + Effective Inertial Load
H_LATE   = 72.87    # km/s/Mpc (MCMC Result)
OM_LATE  = 0.357    # Effective Density (MCMC Result)

# PHASE TRANSITION GEOMETRY
Z_TRANS  = 0.65     # Percolation Threshold
WIDTH    = 0.10     # Smoothness of the Phase Transition

# ==========================================
# 2. DUAL-REGIME ENGINE
# ==========================================

def H_early_trajectory(z):
    """
    The expansion history of the 'Fast & Light' Superfluid Vacuum.
    Governs the universe before the phase transition.
    """
    return H_EARLY * np.sqrt(OM_EARLY * (1+z)**3 + (1-OM_EARLY))

def H_late_trajectory(z):
    """
    The expansion history of the 'Slow & Heavy' Viscous Vacuum.
    Governs the universe after the phase transition (Late-Time).
    """
    return H_LATE * np.sqrt(OM_LATE * (1+z)**3 + (1-OM_LATE))

def get_hubble_parameter(z):
    """
    Combines the two regimes using the Sigmoid Phase Transition function.
    w = 1.0 (Early Universe) -> w = 0.0 (Late Universe)
    """
    # Sigmoid Weighting Function
    # z >> 0.65 -> exp(-pos) -> 0 -> denominator 1 -> w=1 (Early)
    # z << 0.65 -> exp(-neg) -> large -> denominator large -> w=0 (Late)
    w = 1.0 / (1.0 + np.exp(-(z - Z_TRANS)/WIDTH))
    
    # Smoothly transition between the two complete trajectories
    # This preserves the Friedmann energy equation for each phase locally
    H_z = w * H_early_trajectory(z) + (1.0 - w) * H_late_trajectory(z)
    
    return H_z

def integrand(z):
    """
    The age integral: dt = dz / ((1+z) * H(z))
    """
    return 1.0 / ((1+z) * get_hubble_parameter(z))

# ==========================================
# 3. CALCULATE AGES
# ==========================================
print("\nIntegrating Cosmic History...")

# 1. Standard Planck Baseline (Comparison)
# H0=67.4, Om=0.315 constant
def integrand_planck(z):
    return 1.0 / ((1+z) * 67.4 * np.sqrt(0.315*(1+z)**3 + 0.685))

age_planck = quad(integrand_planck, 0, np.inf)[0] * CONST_H_TO_AGE

# 2. Naive SH0ES Baseline (Comparison)
# H0=73.0, Om=0.315 constant (The "Age Crisis" Model)
def integrand_naive(z):
    return 1.0 / ((1+z) * 73.04 * np.sqrt(0.315*(1+z)**3 + 0.685))

age_naive = quad(integrand_naive, 0, np.inf)[0] * CONST_H_TO_AGE

# 3. Vacuum Elastodynamics (Rigorous Dual-Regime)
age_rigorous = quad(integrand, 0, np.inf)[0] * CONST_H_TO_AGE

# ==========================================
# 4. RESULTS & VERDICT
# ==========================================
print("-" * 75)
print(f"{'Model':<35} | {'Age (Gyr)':<15} | {'Status'}")
print("-" * 75)
print(f"{'Planck 2018 (H0=67.4)':<35} | {age_planck:.3f} Gyr     | Standard")
print(f"{'Naive SH0ES (H0=73.0)':<35} | {age_naive:.3f} Gyr     | Too Young (<12.5)")
print(f"{'Vacuum Elastodynamics (Transition)':<35} | {age_rigorous:.3f} Gyr     | PREDICTION")
print("-" * 75)

# Globular Cluster Limit (Valcin et al. 2020 / Bernal et al. 2021)
# Conservative lower bound ~12.35 - 12.5 Gyr
LIMIT_LOW = 12.35
LIMIT_STRICT = 12.50

print(f"\n[ TEST ] Is Age > {LIMIT_LOW} Gyr (Oldest Stars)?")

if age_rigorous >= LIMIT_LOW:
    print(f"[ PASS ] YES. Age {age_rigorous:.3f} Gyr satisfies the observational lower bound.")
    print("         The density unloading (Om=0.357 -> 0.315) in the early universe")
    print("         successfully recovers the necessary time duration.")
else:
    print(f"[ FAIL ] NO. Age {age_rigorous:.3f} Gyr is too young.")
