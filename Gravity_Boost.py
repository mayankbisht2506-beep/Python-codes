import numpy as np
from scipy.integrate import quad
from scipy.optimize import root_scalar

print("--- TEST 1: EXACT TOPOLOGICAL ROOT (PURE GEOMETRY) ---")
print("Objective: Derive H_fast and H_local strictly from exact Friedmann Integrals")

# ==========================================
# 1. THE EXACT PLANCK 2018 BASELINE
# ==========================================
# We use this ONLY to find the baseline Comoving Distance 
# that perfectly locks the CMB angle for the standard model.
H0_LCDM = 67.36
OM_LCDM = 0.3153
C_LIGHT = 299792.458

# ==========================================
# 2. THEORETICAL INPUTS (Strictly from E8 Topology)
# ==========================================
CABIBBO_ANGLE = 0.2250      # D4 Triality projection
Y_MAX = 0.2055              # Geometric Yield Coefficient
OM_BARE = 0.3116            # Topological Percolation Density
ZETA_SAT = 0.1569           # Lepton Saturation Viscosity

# ==========================================
# 3. EXACT CALCULATION CASCADE
# ==========================================
# A. Gravity Boost
DELTA_EFF = CABIBBO_ANGLE * (1.0 - Y_MAX)
G_BOOST = 1.0 / (1.0 - DELTA_EFF)

# B. Native Baseline Comoving Distance
def get_comoving_distance(h_val, om_val):
    integrand = lambda z: 1.0 / (h_val * np.sqrt(om_val*(1+z)**3 + (1-om_val)))
    return C_LIGHT * quad(integrand, 0, 1090.0)[0]

da_standard = get_comoving_distance(H0_LCDM, OM_LCDM)

# C. The Absolute Target Comoving Distance 
# (Shrinks perfectly by the geometric invariant to preserve theta_*)
da_target_vacuum = da_standard / np.sqrt(G_BOOST)

# D. Step 1: The Fast Early Trajectory (Superfluid Ceiling)
# Use a high-precision continuous root solver to find exact H_fast
def objective_function(h_guess):
    return get_comoving_distance(h_guess, OM_BARE) - da_target_vacuum

res = root_scalar(objective_function, bracket=[70, 80])
H_FAST_EXACT = res.root

# E. Step 2: Phase Transition Drag
DELTA_OM = ZETA_SAT / 3.0   
B_VISC = np.sqrt(1.0 - DELTA_OM)

# F. Step 3: The Decelerated Local Trajectory (Terminal Velocity)
H0_LOCAL_PREDICTED = H_FAST_EXACT * B_VISC

# ==========================================
# 4. RESULTS
# ==========================================
print("-" * 65)
print(f"Topological Bare Density:        {OM_BARE:.4f}")
print(f"Gravity Boost Factor:            {G_BOOST:.4f}")
print(f"LCDM Baseline Comoving Dist:     {da_standard:.2f} Mpc")
print(f"Vacuum Target Comoving Dist:     {da_target_vacuum:.2f} Mpc")
print("-" * 65)
print(f"Exact Theoretical Ceiling:       {H_FAST_EXACT:.6f} km/s/Mpc")
print(f"Inertial Counter-Load (Drag):   -{(1 - B_VISC)*100:.2f}%")
print(f"Exact Terminal Velocity:         {H0_LOCAL_PREDICTED:.6f} km/s/Mpc")
print("-" * 65)
