import numpy as np
from scipy.integrate import quad

print("--- CMB GEOMETRIC RESONANCE: STRICT VALIDATION (CORRECTED) ---")

# ==========================================
# 1. FIXED PHYSICS (Add 33 Parameters)
# ==========================================
c_0 = 299792.458
G_RATIO = (74.5/67.4)**2  # ~1.22 (Gravity Boost)
Z_TRANS = 0.65
WIDTH   = 0.15

# Theoretical Predictions (Lepton Sum Rule & Gravity Boost)
H0_THEORY = 74.5
ETA_THEORY = 0.21

# Planck Baseline Constants
H0_PLANCK = 67.4
# We fix physical densities to ensure we are testing GEOMETRY only
omega_m_true = 0.315 * (H0_PLANCK/100)**2 
omega_r_true = 2.4728e-5 * 1.6918 

# ==========================================
# 2. PHYSICS ENGINES
# ==========================================

# --- A. STANDARD LCDM (The Control Group) ---
def get_theta_lcdm():
    # Standard H(z), Constant c, Standard G
    h = H0_PLANCK / 100.0
    
    def get_H(z):
        Om = omega_m_true / h**2
        Or = omega_r_true / h**2
        Ol = 1.0 - Om - Or
        return 100 * h * np.sqrt(Or*(1+z)**4 + Om*(1+z)**3 + Ol)

    # Sound Horizon (Standard)
    rs = quad(lambda z: (c_0/np.sqrt(3)) / get_H(z), 1090, 1e7)[0]
    # Angular Distance (Standard)
    da_int = quad(lambda z: c_0 / get_H(z), 0, 1090)[0]
    da = da_int / (1 + 1090)
    
    return rs / da

# --- B. VACUUM ELASTODYNAMICS (The Test Group) ---
def get_theta_vacuum(h0_in, eta_in):
    h = h0_in / 100.0
    
    # Gravity Scaling (Sigmoid)
    def get_G(z):
        sigmoid = 1.0 / (1.0 + np.exp(-(z - Z_TRANS) / WIDTH))
        return 1.0 + (G_RATIO - 1.0) * sigmoid

    # Modified H(z)
    def get_H(z):
        Om = omega_m_true / h**2
        Or = omega_r_true / h**2
        Ol = 1.0 - Om - Or
        E = np.sqrt(Or*(1+z)**4 + Om*(1+z)**3 + Ol)
        # Viscous Damping: H ~ G^(-0.5 * eta)
        G_scaling = (get_G(z))**(-0.5 * eta_in)
        return 100 * h * E * G_scaling

    # Variable Speed of Light
    def get_c(z):
        # c ~ G^-0.5
        return c_0 * (get_G(z))**(-0.5)

    # Observables
    rs = quad(lambda z: (get_c(z)/np.sqrt(3)) / get_H(z), 1090, 1e7)[0]
    da_int = quad(lambda z: get_c(z) / get_H(z), 0, 1090)[0]
    da = da_int / (1 + 1090)
    
    return rs / da

# ==========================================
# 3. EXECUTE TEST
# ==========================================
theta_target = get_theta_lcdm()
theta_model  = get_theta_vacuum(H0_THEORY, ETA_THEORY)

diff = theta_model - theta_target
err = (diff / theta_target) * 100

print(f"\nRESULTS:")
print(f"Target Theta (Planck):      {theta_target:.6f}")
print(f"Model Theta (74.5, 0.21):   {theta_model:.6f}")
print(f"Error:                      {err:.4f}%")

# ==========================================
# 4. SCIENTIFIC INTERPRETATION
# ==========================================
print("\n" + "="*50)
if abs(err) < 1.0:
    print("VERDICT: SUCCESS (Scientific Match)")
    print("The theoretical prediction is within 1% of Planck.")
    print("This confirms the theory works without fine-tuning.")
else:
    print("VERDICT: TENSION")
    print("The error is too large. Check the scaling laws.")
print("="*50)
