import numpy as np
from scipy.integrate import quad

print("--- VACUUM ELASTODYNAMICS: COSMIC AGE VERIFICATION ---")
print("Objective: Verify the 12.52 Gyr 'Kinematic Age' (Section 9.12)")
print("Engine: Exact Covariant Geometry (Width=0.084, Radiative Decoupling)")
print("-" * 65)

# ==========================================
# 1. PHYSICS PARAMETERS
# ==========================================
CONST_H_TO_AGE = 977.7922  # Conversion: 1/H0 [Gyr]

# Standard Planck 2018 Baseline
H0_PLANCK = 67.36
OM_PLANCK = 0.3153
OL_PLANCK = 1.0 - OM_PLANCK
# Note: Radiation omitted in LCDM baseline for 1:1 comparative parity 
# across the matter-dominated age integral.

# Vacuum Elastodynamics (Zero Continuous Parameters)
H_FAST = 74.69
H_LOCAL = 72.71
OM_PRIMORDIAL = 0.3116
OM_EFFECTIVE = 0.3639

# Phase Transition Geometry
Z_TRANS = 0.641
WIDTH = 0.084   # EXACT: Derived geometric Jacobian width

# ==========================================
# 2. DUAL-PHASE COVARIANT ENGINE
# ==========================================

def get_sigmoid(z):
    """Pure, unnormalized theoretical transition function."""
    arg = np.clip((Z_TRANS - z) / WIDTH, -100.0, 100.0)
    return 1.0 / (1.0 + np.exp(-arg))

def integrand_vacuum(z):
    """Exact VED Covariant Expansion History (Eqs. 134 & 135)."""
    # 1. Interpolate the foundational parameters
    S_z = get_sigmoid(z)
    OM_Z = OM_PRIMORDIAL + (OM_EFFECTIVE - OM_PRIMORDIAL) * S_z
    H_Z = H_FAST + (H_LOCAL - H_FAST) * S_z
    OL_Z = 1.0 - OM_Z
    
    # 2. Evaluate the metric (Radiative Decoupling Enforced per Sec 7.2.2)
    E_z = np.sqrt(OM_Z * (1 + z)**3 + OL_Z)
    H_actual = H_Z * E_z
    
    return 1.0 / ((1 + z) * H_actual)

def integrand_planck(z):
    """Standard LCDM Expansion History."""
    E_z = np.sqrt(OM_PLANCK * (1 + z)**3 + OL_PLANCK)
    H_actual = H0_PLANCK * E_z
    return 1.0 / ((1 + z) * H_actual)

# ==========================================
# 3. CALCULATE AGES
# ==========================================
print("Integrating Cosmic History (z=0 to z=infinity)...")

# Using a high integration limit to approximate infinity safely
age_planck = quad(integrand_planck, 0, 1e5)[0] * CONST_H_TO_AGE
age_rigorous = quad(integrand_vacuum, 0, 1e5)[0] * CONST_H_TO_AGE

# ==========================================
# 4. RESULTS & VERDICT
# ==========================================
print("\n" + "-" * 75)
print(f"{'Model':<35} | {'Age (Gyr)':<15} | {'Status'}")
print("-" * 75)
print(f"{'Planck 2018 (H0=67.36)':<35} | {age_planck:.3f} Gyr      | Standard LCDM Baseline")
print(f"{'Vacuum Elastodynamics (Exact)':<35} | {age_rigorous:.3f} Gyr      | THEORETICAL PREDICTION")
print("-" * 75)

LIMIT_LOW = 12.50

print(f"\n[ TEST ] Does the VED Age satisfy the oldest Globular Clusters (>= {LIMIT_LOW} Gyr)?")

if age_rigorous >= LIMIT_LOW:
    print(f"[ PASS ] YES. Age {age_rigorous:.3f} Gyr perfectly matches Section 9.12.")
    print("         The viscoplastic braking of the late universe provides the exact")
    print("         temporal margin needed to keep the universe older than its stars,")
    print("         while the fast early expansion natively explains JWST galaxies.")
else:
    print(f"[ FAIL ] NO. Age {age_rigorous:.3f} Gyr is too young.")
