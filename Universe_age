import numpy as np
from scipy.integrate import quad

# ==========================================
# 1. PARAMETERS (VALIDATED)
# ==========================================
CONST_H_TO_AGE = 977.8 

# [span_2](start_span)Standard LCDM[span_2](end_span)
H0_PLANCK = 67.4
OM_PLANCK = 0.315

# [span_3](start_span)[span_4](start_span)Vacuum Model[span_3](end_span)[span_4](end_span)
H0_VAC = 73.0       # Late Universe H0
OM_VAC = 0.315      # Assumed similar for shape consistency
Z_TRANS = 0.65      # Transition Redshift
WIDTH = 0.1         # Smoothness (Implied by Fig 2)

# ==========================================
# 2. PHYSICS FUNCTIONS (TRANSITION MODEL)
# ==========================================
def E(z):
    # Standard expansion kernel
    return np.sqrt(OM_PLANCK*(1+z)**3 + (1-OM_PLANCK))

def H_effective(z):
    """
    Model: Effective H0 relaxes from 73 (z=0) to ~67.4 (z>0.65).
    Ref: Section 7.1, Figure 2 "Effective Hubble Constant relaxes..."
    """
    # Sigmoid weighting: 1.0 at z=0, 0.0 at high z
    # Note: Logic inverted from G. H is HIGH at low z, LOW at high z.
    weight = 1.0 / (1.0 + np.exp((z - Z_TRANS)/WIDTH))
    
    # Smooth transition between H0_VAC and H0_PLANCK
    H0_eff = H0_PLANCK + (H0_VAC - H0_PLANCK) * weight
    
    return H0_eff * E(z)

def integrand(z):
    return 1.0 / ( (1+z) * H_effective(z) )

# ==========================================
# 3. CALCULATE AGES
# ==========================================
# Standard
age_planck = quad(lambda z: 1/((1+z)*H0_PLANCK*E(z)), 0, np.inf)[0] * CONST_H_TO_AGE

# Naive H0=73 (Constant 73 everywhere)
age_naive = quad(lambda z: 1/((1+z)*H0_VAC*E(z)), 0, np.inf)[0] * CONST_H_TO_AGE

# Vacuum Model (Transition Integration)
age_vac = quad(integrand, 0, np.inf)[0] * CONST_H_TO_AGE

print(f"--- COSMIC CHRONOMETER TEST ---")
print(f"Planck (H0=67.4):        {age_planck:.2f} Gyr")
print(f"Naive H0=73.0 (Risk):    {age_naive:.2f} Gyr (Too Young)")
print(f"Vacuum Model (Fig 2):    {age_vac:.2f} Gyr (Paper: 13.05 Gyr)")

# ==========================================
# 4. VERDICT
# ==========================================
# [span_5](start_span)Paper Reference[span_5](end_span): 
# "consistent with the lower limit set by globular cluster ages (~12.5 Gyr)"
if age_vac > 12.5:
    print("VERDICT: PASS. Consistent with Globular Clusters (>12.5 Gyr).")
else:
    print("VERDICT: FAIL. Too young for stars.")

