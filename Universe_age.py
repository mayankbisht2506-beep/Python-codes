import numpy as np
from scipy.integrate import quad

print("--- COSMIC AGE TEST: STRICT LEPTON LIMIT ---")
print("Objective: Verify if H0=74.5 (Strict Prediction) violates the Age of the Universe.")

# ==========================================
# 1. PARAMETERS (STRICT THEORETICAL VALUES)
# ==========================================
CONST_H_TO_AGE = 977.8 

# Standard LCDM (Planck Baseline)
H0_PLANCK = 67.4
OM_PLANCK = 0.315

# VACUUM MODEL (STRICT LEPTON RULE)
# Reference: Section 7.1 & 9.12
# "This stiffer vacuum predicts a local Hubble constant of H0 ~ 74.5"
H0_THEORY = 74.5    # STRICT PREDICTION (Not 73.0)
Z_TRANS   = 0.65    # Percolation Threshold
WIDTH     = 0.1     

def E(z):
    return np.sqrt(OM_PLANCK*(1+z)**3 + (1-OM_PLANCK))

def H_effective(z):
    """
    Strict Model:
    The universe expands at H0=74.5 today (z=0), but relaxes to 
    the Planck rate (H0=67.4) at high redshifts (z > 0.65).
    """
    # Sigmoid Transition
    weight = 1.0 / (1.0 + np.exp((z - Z_TRANS)/WIDTH))
    
    # Smooth transition from 74.5 (Today) -> 67.4 (Early)
    H0_eff = H0_PLANCK + (H0_THEORY - H0_PLANCK) * weight
    
    return H0_eff * E(z)

def integrand(z):
    return 1.0 / ( (1+z) * H_effective(z) )

# ==========================================
# 2. CALCULATE AGES
# ==========================================
# 1. Planck LCDM (H0 = 67.4 everywhere)
age_planck = quad(lambda z: 1/((1+z)*H0_PLANCK*E(z)), 0, np.inf)[0] * CONST_H_TO_AGE

# 2. Naive High-H0 (H0 = 74.5 everywhere)
# This usually breaks the age limit (< 12.5 Gyr)
age_naive = quad(lambda z: 1/((1+z)*H0_THEORY*E(z)), 0, np.inf)[0] * CONST_H_TO_AGE

# 3. Vacuum Elastodynamics (H0 = 74.5 -> 67.4)
age_vac = quad(integrand, 0, np.inf)[0] * CONST_H_TO_AGE

# ==========================================
# 3. RESULTS
# ==========================================
print("-" * 60)
print(f"{'Model':<25} | {'H0 Value':<10} | {'Calculated Age'}")
print("-" * 60)
print(f"{'Planck (Standard)':<25} | {H0_PLANCK:<10} | {age_planck:.2f} Gyr")
print(f"{'Naive (Constant H0)':<25} | {H0_THEORY:<10} | {age_naive:.2f} Gyr (CRITICAL FAILURE)")
print(f"{'Vacuum (Strict Lepton)':<25} | {H0_THEORY:<10} | {age_vac:.2f} Gyr")
print("-" * 60)

# ==========================================
# 4. SCIENTIFIC VERDICT
# ==========================================
# Globular Cluster Limit: ~12.5 Gyr (Cimatti et al. 2019)
limit = 12.5

print(f"\nTEST: Is Age > {limit} Gyr?")
if age_vac > limit:
    print("VERDICT: PASS.")
    print("Even with H0=74.5, the Phase Transition preserves the cosmic age.")
    print("The universe is older than 12.5 Gyr because H(z) was lower in the past.")
else:
    print(f"VERDICT: FAIL. Age {age_vac:.2f} Gyr is too young.")
