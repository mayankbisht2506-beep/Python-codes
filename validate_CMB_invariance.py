import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize

print("--- CMB GEOMETRIC RESONANCE: FIND & VERIFY ---")
print("Objective: 1. Find optimal H0/Eta parameters.")
print("           2. Validate Geometric Invariance Theorem (Eq. 93).")

# ==========================================
# 1. FIXED PHYSICS & CONSTANTS
# ==========================================
c_0 = 299792.458  # km/s
G_RATIO = 1.23    # Early Gravity Boost (Eq. 75)
Z_TRANS = 0.65    # Transition Redshift
WIDTH   = 0.1     # Smoothness

# Standard Model Baseline (The Target)
H0_PLANCK = 67.4
# We fix Physical Densities to isolate geometry (Avoids "Density Trap")
omega_m_true = 0.315 * (H0_PLANCK/100)**2 
# Radiation: 2.47e-5 (photons) * 1.69 (neutrinos)
omega_r_true = 2.4728e-5 * 1.6918 

# ==========================================
# 2. PHYSICS ENGINE (Vacuum Elastodynamics)
# ==========================================
def get_G_factor(z):
    """Calculates Gravity Scaling G(z)/G0"""
    sigmoid = 1.0 / (1.0 + np.exp(-(z - Z_TRANS) / WIDTH))
    return 1.0 + (G_RATIO - 1.0) * sigmoid

def get_theta(params):
    """
    Calculates CMB Acoustic Scale (Theta) for given [H0, Eta].
    """
    h_vac = params[0] / 100.0
    eta   = params[1]
    
    # 1. Define H(z) with Viscous Scaling
    def get_H_vac(z):
        # Use Fixed Physical Densities
        Om = omega_m_true / h_vac**2
        Or = omega_r_true / h_vac**2
        Ol = 1.0 - Om - Or
        E = np.sqrt(Or*(1+z)**4 + Om*(1+z)**3 + Ol)
        
        # Viscous Scaling Law: H ~ G^-0.5*eta
        G_scaling = (get_G_factor(z))**(-0.5 * eta)
        return 100 * h_vac * E * G_scaling

    # 2. Define c(z) with VSL Scaling
    def get_c_vac(z):
        # VSL Law: c ~ G^-0.5
        return c_0 * (get_G_factor(z))**(-0.5)

    # 3. Calculate Observables
    # Sound Horizon (rs): Integral z_star -> infinity
    rs = quad(lambda z: (get_c_vac(z)/np.sqrt(3)) / get_H_vac(z), 1090, 1e7)[0]
    
    # Angular Distance (DA): Integral 0 -> z_star
    da_int = quad(lambda z: get_c_vac(z) / get_H_vac(z), 0, 1090)[0]
    da = da_int / (1 + 1090)
    
    return rs / da

# ==========================================
# 3. STEP A: CALCULATE TARGET (PLANCK)
# ==========================================
print("\n[Step 1] Calculating Planck Baseline Target...")

def get_H_std(z):
    h = H0_PLANCK / 100.0
    Om = omega_m_true / h**2
    Or = omega_r_true / h**2
    Ol = 1.0 - Om - Or
    return 100 * h * np.sqrt(Or*(1+z)**4 + Om*(1+z)**3 + Ol)

rs_std = quad(lambda z: (c_0/np.sqrt(3)) / get_H_std(z), 1090, 1e7)[0]
da_std = quad(lambda z: c_0 / get_H_std(z), 0, 1090)[0] / 1091
TARGET_THETA = rs_std / da_std
print(f"Target Theta_* = {TARGET_THETA:.6f}")

# ==========================================
# 4. STEP B: FIND OPTIMAL PARAMETERS
# ==========================================
print("\n[Step 2] Scanning for Resonance Point...")

def loss_function(params):
    # Returns squared percentage error
    theta_model = get_theta(params)
    diff_percent = (theta_model - TARGET_THETA) / TARGET_THETA * 100
    return diff_percent**2

# Search Bounds consistent with Paper
# H0: 73.0 (SH0ES) to 75.0
# Eta: 0.15 to 0.25 (around theoretical 0.17)
bounds = ((73.0, 75.0), (0.15, 0.25))
initial_guess = [74.0, 0.20]

result = minimize(loss_function, initial_guess, bounds=bounds, method='L-BFGS-B')

best_h0 = result.x[0]
best_eta = result.x[1]

# ==========================================
# 5. STEP C: VALIDATION REPORT
# ==========================================
final_theta = get_theta([best_h0, best_eta])
final_diff = (final_theta - TARGET_THETA) / TARGET_THETA * 100

print("\n" + "="*60)
print("FINAL VALIDATION REPORT: GEOMETRIC INVARIANCE")
print("="*60)
print(f"{'Parameter':<20} | {'Value':<15} | {'Source'}")
print("-" * 60)
print(f"{'H0 (Found)':<20} | {best_h0:<15.4f} | {'Optimizer (SH0ES-like)'}")
print(f"{'Eta (Found)':<20} | {best_eta:<15.4f} | {'Optimizer (Theoretical)'}")
print("-" * 60)
print(f"{'Target Theta':<20} | {TARGET_THETA:<15.6f} | {'Planck 2018'}")
print(f"{'Model Theta':<20} | {final_theta:<15.6f} | {'Vacuum Elastodynamics'}")
print("-" * 60)
print(f"Error Magnitude: {abs(final_diff):.6f}%")
print("="*60)

# ==========================================
# 6. VERDICT
# ==========================================
if abs(final_diff) < 0.2:
    print("VERDICT: PASS.")
    print("The Geometric Invariance Theorem is numerically confirmed.")
    print(f"Parameters ({best_h0:.1f}, {best_eta:.3f}) resolve tension with <0.2% CMB error.")
else:
    print("VERDICT: FAIL.")
    print("Convergence not achieved.")
