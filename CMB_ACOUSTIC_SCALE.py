import numpy as np
from scipy.integrate import quad

# ==========================================
# 1. PHYSICAL CONSTANTS (ABSOLUTE UNITS)
# ==========================================
c = 299792.458  # km/s
# Planck 2018 Best Fits (Physical Densities)
# These are the "Anchors" of cosmology
omega_b = 0.0224  # Baryon density
omega_c = 0.1200  # Dark Matter density
omega_r = 4.15e-5 # Radiation density (fixed by T_cmb)
# Derived parameters
omega_m = omega_b + omega_c

# ==========================================
# 2. PHYSICS MODELS
# ==========================================
def hubble_rate(z, model='std'):
    """
    Returns H(z) in km/s/Mpc based on physical densities.
    Friedmann: H^2 = (8piG/3) * rho
    We scale relative to standard G.
    """
    # Standard Physical Expansion Rate (H^2 / 100^2)
    # E2 = omega_r(1+z)^4 + omega_m(1+z)^3 + omega_L
    # But omega_L depends on H0 closure.

    if model == 'std':
        h = 0.674     # Planck H0
        G_factor = 1.0
        # Dark Energy for flatness
        omega_L = h**2 - omega_m - omega_r
    else:
        # Vacuum Model
        h = 0.733     # SH0ES H0
        # Gravity Logic:
        # Early U (z > 0.65): G = 1.23 * G0
        # Late U  (z < 0.65): G = 1.00 * G0
        if z > 0.65:
            G_factor = 1.23
        else:
            G_factor = 1.0

        # For Vacuum Model, H0 is determined by late-time sum
        # We enforce H0=73.3 at z=0
        omega_L = h**2 - omega_m - omega_r

    # Calculate H(z)^2 component from matter/radiation
    density_term = omega_r*(1+z)**4 + omega_m*(1+z)**3

    # Apply Gravity Boost to density sourcing
    H2_density = G_factor * density_term

    # Add Dark Energy (Vacuum Energy constant in late universe)
    # Assumption: DE dominates late, unaffected by G_boost scaling of matter
    H2 = H2_density + omega_L

    return 100.0 * np.sqrt(H2)

# ==========================================
# 3. SOUND HORIZON (r_s)
# ==========================================
def get_rs(model='std'):
    # Integration from Recombination (z=1090) to Infinity
    z_star = 1090.0

    # Sound speed (simplified approx c/sqrt(3))
    # In full physics, depends on baryon loading, but ratio test holds.
    cs = c / np.sqrt(3)

    # Integrand: cs / H(z)
    def integrand(z):
        return cs / hubble_rate(z, model)

    r_s, _ = quad(integrand, z_star, 1e7)
    return r_s

# ==========================================
# 4. ANGULAR DIAMETER DISTANCE (D_A)
# ==========================================
def get_da(model='std'):
    z_star = 1090.0

    # Integrand: c / H(z)
    def integrand(z):
        return c / hubble_rate(z, model)

    # Proper Distance
    r_prop, _ = quad(integrand, 0, z_star)

    # Angular Diameter Distance = r / (1+z)
    return r_prop / (1 + z_star)

# ==========================================
# 5. EXECUTE TEST
# ==========================================
# Standard LCDM
rs_std = get_rs('std')
da_std = get_da('std')
theta_std = rs_std / da_std

# Vacuum Elastodynamics
rs_vac = get_rs('vac')
da_vac = get_da('vac')
theta_vac = rs_vac / da_vac

# Comparison
diff = abs(theta_vac - theta_std) / theta_std * 100

print(f"--- CMB ACOUSTIC SCALE CONSISTENCY ---")
print(f"Standard Model:")
print(f"  r_s = {rs_std:.1f} Mpc")
print(f"  D_A = {da_std:.1f} Mpc")
print(f"  Theta_* = {theta_std:.6f} rad")
print(f"Vacuum Model:")
print(f"  r_s = {rs_vac:.1f} Mpc (Contracted)")
print(f"  D_A = {da_vac:.1f} Mpc (Contracted)")
print(f"  Theta_* = {theta_vac:.6f} rad")
print(f"-"*30)
print(f"Difference: {diff:.3f}%")

if diff < 1.0:
    print("VERDICT: PASS (Peaks Preserved)")
else:
    print("VERDICT: FAIL (Adjust G_BOOST or H0)")
