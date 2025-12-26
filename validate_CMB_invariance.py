import numpy as np
from scipy.integrate import quad

print("--- VSL-VISCOSITY CMB TEST (PAPER COMPLIANT) ---")
print("Objective: Validate Eq. 93 (Geometric Cancellation) using VSL scaling.")

# ==========================================
# 1. CONSTANTS & PARAMETERS
# ==========================================
c_0 = 299792.458  # km/s (Local value)

# From Paper Section 7.1 & 7.4
Z_TRANS = 0.65    # 
WIDTH   = 0.1     # Smooth transition
ETA     = 0.17    # Viscosity 
G_RATIO = 1.23    # G_early / G_0 

# Cosmology (Planck 2018 Baseline) [cite: 5369]
Om = 0.315
Or = 4.15e-5      # Radiation
Ol = 1.0 - Om - Or
H0_PLANCK = 67.4  # [cite: 3622]
H0_VAC    = 73.0  # [cite: 3611]

# ==========================================
# 2. PHYSICS FUNCTIONS (VSL IMPLEMENTATION)
# ==========================================
def get_G_factor(z):
    """
    Returns G(z)/G_0 based on Stiffness Phase Transition.
    Eq. 71: G relaxes from 1.23 to 1.0
    """
    # Sigmoid function for phase transition
    sigmoid = 1.0 / (1.0 + np.exp(-(z - Z_TRANS) / WIDTH))
    # Early (z>>0.65) -> sigmoid=1 -> G=1.23
    # Late  (z<<0.65) -> sigmoid=0 -> G=1.0
    return 1.0 + (G_RATIO - 1.0) * sigmoid

def get_c_z(z):
    """
    Variable Speed of Light Scaling.
    Eq. 89: c(z) propto G(z)^-0.5 [cite: 4502]
    """
    G_fact = get_G_factor(z)
    return c_0 * (G_fact)**(-0.5)

def get_H_z(z, model='std'):
    """
    Expansion Rate H(z).
    Standard: Friedmann
    Vacuum: Viscous Drag Scaling 
    """
    E_std = np.sqrt(Or*(1+z)**4 + Om*(1+z)**3 + Ol)
    
    if model == 'std':
        return H0_PLANCK * E_std
    else:
        # Vacuum Model: H0 = 73.0 locally
        # Scaling: H(z) propto G(z)^-0.5(1-eta)
        G_fact = get_G_factor(z)
        
        # Calculate the Viscous Scaling Factor relative to z=0
        # Factor = (G(z)/G(0)) ^ (-0.5 * (1 - eta))
        scaling = (G_fact / 1.0)**(-0.5 * (1.0 - ETA))
        
        return H0_VAC * E_std * scaling

def get_sound_speed(z, model='std'):
    """
    Sound speed c_s(z).
    Scales with c(z). Uses approx c/sqrt(3) for geometric ratio.
    """
    if model == 'std':
        c_local = c_0
    else:
        c_local = get_c_z(z) # VSL active 
        
    return c_local / np.sqrt(3)

# ==========================================
# 3. GEOMETRIC INTEGRATION
# ==========================================
z_star = 1090.0 # Recombination

def calc_observables(model):
    # A. Sound Horizon (r_s)
    # Integral of c_s(z) / H(z)
    def integ_rs(z):
        return get_sound_speed(z, model) / get_H_z(z, model)
    rs, _ = quad(integ_rs, z_star, 1e7)
    
    # B. Angular Diameter Distance (D_A)
    # Integral of c(z) / H(z) ... (Wait, DA uses c(z) in VSL?)
    # Standard formula: DA = 1/(1+z) * Integral(c/H)
    # In VSL, the 'c' in the metric integral is c(z).
    def integ_da(z):
        if model == 'std': return c_0 / get_H_z(z, model)
        else: return get_c_z(z) / get_H_z(z, model) # 
        
    r_comoving, _ = quad(integ_da, 0, z_star)
    da = r_comoving / (1 + z_star)
    
    return rs, da

# ==========================================
# 4. EXECUTE & VERIFY
# ==========================================
print("Calculating Standard Model...")
rs_std, da_std = calc_observables('std')
theta_std = rs_std / da_std

print("Calculating Vacuum Elastodynamics (VSL + Viscosity)...")
rs_vac, da_vac = calc_observables('vac')
theta_vac = rs_vac / da_vac

diff = (theta_vac - theta_std) / theta_std * 100

print("\n" + "="*50)
print("RESULTS: CMB ACOUSTIC SCALE (Theta_*)")
print(f"Paper Reference: Section 7.8, Eq. 93 ")
print("="*50)
print(f"Standard (H0={H0_PLANCK}):")
print(f"  r_s   = {rs_std:.2f} Mpc")
print(f"  D_A   = {da_std:.2f} Mpc")
print(f"  Theta = {theta_std:.6f} rad")
print("-" * 50)
print(f"Vacuum (H0={H0_VAC}, eta={ETA}, G_early={G_RATIO}x):")
print(f"  r_s   = {rs_vac:.2f} Mpc (Viscous Contraction)")
print(f"  D_A   = {da_vac:.2f} Mpc")
print(f"  Theta = {theta_vac:.6f} rad")
print("-" * 50)
print(f"Difference: {diff:.3f}%")
print("="*50)

if abs(diff) < 1.0:
    print("VERDICT: PASS. Matches Eq. 93 prediction (<1%).")
else:
    print("VERDICT: FAIL. Check scaling relations.")
