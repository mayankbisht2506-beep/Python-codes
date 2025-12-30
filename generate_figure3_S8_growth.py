import numpy as np
from scipy.integrate import odeint, quad
import matplotlib.pyplot as plt

# ==========================================
# 1. PHYSICAL CONSTANTS
# ==========================================
H0_PLANCK = 67.4
OM_PLANCK = 0.315
S8_PLANCK = 0.832

# Observational Targets
H0_SHOES = 73.04
S8_DES = 0.776
S8_KIDS = 0.759

# --- LEPTON GEOMETRIC MODEL PARAMETERS ---
ETA_FLOOR = 0.21       # Lepton Sum Rule (Beta/2)
DELTA_GEO = 0.229      # E8 Lattice Geometric Limit
ETA_PEAK = 0.31        # Jamming Transition Peak
Z_TRANS = 0.65         # Percolation Threshold

# ==========================================
# 2. HUBBLE TENSION ENGINE (Geometric Model)
# ==========================================
def calculate_hubble_geometric(eta):
    """
    Calculates H0 using the Geometric Relaxation Formula:
    G_boost = 1 / (1 - delta_geo * (1 - eta))
    """
    # 1. Effective Relaxation (Modulated by Viscosity)
    delta_eff = DELTA_GEO * (1.0 - eta)

    # 2. Gravity Boost (Stiffness Relaxation)
    g_boost = 1.0 / (1.0 - delta_eff)

    # 3. Hubble Boost (H ~ sqrt(G))
    h_boost = np.sqrt(g_boost)

    # 4. Prediction
    h0_pred = H0_PLANCK * h_boost
    mag_shift = -5 * np.log10(h0_pred / H0_PLANCK)

    return h0_pred, mag_shift, g_boost

# ==========================================
# 3. S8 TENSION ENGINE (Quadratic Impedance)
# ==========================================
def get_viscosity_profile(z):
    # Sigmoid activation for late-time floor
    width = 0.1
    arg = (z - Z_TRANS) / width
    late_trigger = 1.0 / (1.0 + np.exp(arg))

    # Base Floor (0.21)
    base = ETA_FLOOR * late_trigger

    # Jamming Spike (Gaussian at 0.65)
    spike = (ETA_PEAK - ETA_FLOOR) * np.exp(-0.5 * ((z - Z_TRANS)/0.15)**2)

    return base + spike

def growth_ode(y, a, model='lcdm'):
    delta, delta_prime = y
    z = 1.0/a - 1.0

    E = np.sqrt(OM_PLANCK*(1+z)**3 + (1-OM_PLANCK))
    dE_da = -1.5 * OM_PLANCK * (a**-4) / E
    hubble_friction = 3.0/a + dE_da/E
    gravity_source = 1.5 * OM_PLANCK / (a**5 * E**2)

    if model == 'viscous':
        eta = get_viscosity_profile(z)
        # QUADRATIC IMPEDANCE LAW: Friction ~ (1 + eta)^2
        friction_term = hubble_friction * (1.0 + eta)**2.0
    else:
        friction_term = hubble_friction

    return [delta_prime, -friction_term*delta_prime + gravity_source*delta]

# ==========================================
# 4. EXECUTION
# ==========================================
print(f"--- LEPTON GEOMETRIC MODEL DIAGNOSTIC ---")
print(f"Viscosity Input: eta = {ETA_FLOOR} (Lepton Sum Rule)")

# A. Run Hubble Check
h0_pred, mag_shift, g_boost = calculate_hubble_geometric(ETA_FLOOR)
print(f"\n[1] HUBBLE TENSION (Geometric Model)")
print(f"   Effective Relaxation: {DELTA_GEO * (1-ETA_FLOOR):.4f}")
print(f"   Gravity Boost:        {g_boost:.4f}x")
print(f"   Predicted H0:         {h0_pred:.2f} km/s/Mpc")
print(f"   Magnitude Shift:      {mag_shift:.4f} mag")
print(f"   Target (SH0ES):       {H0_SHOES:.2f}")
print(f"   Status:               {'SOLVED' if 73.0 < h0_pred < 76.0 else 'FAIL'}")

# B. Run S8 Check
print(f"\n[2] S8 CLUSTERING (Quadratic Impedance)")
a_range = np.linspace(0.001, 1.0, 500)
y0 = [a_range[0], 1.0]

sol_lcdm = odeint(growth_ode, y0, a_range, args=('lcdm',))
sol_visc = odeint(growth_ode, y0, a_range, args=('viscous',))

growth_suppression = sol_visc[-1,0] / sol_lcdm[-1,0]
s8_pred = S8_PLANCK * growth_suppression

print(f"   Growth Suppression:   {growth_suppression:.4f}")
print(f"   Predicted S8:         {s8_pred:.3f}")
print(f"   Target (KiDS/DES):    {S8_KIDS} - {S8_DES}")

if 0.75 <= s8_pred <= 0.78:
    print(f"   Status:               PERFECT RESOLUTION")
elif s8_pred < 0.75:
    print(f"   Status:               OVERSHOOT (Too much damping)")
else:
    print(f"   Status:               UNDERSHOOT (Not enough damping)")

# C. Visual Confirmation
plt.figure(figsize=(10,5))
z_axis = 1/a_range - 1
plt.plot(z_axis, sol_lcdm[:,0]/sol_lcdm[-1,0], 'k--', label='Standard LCDM')
plt.plot(z_axis, sol_visc[:,0]/sol_lcdm[-1,0], 'r-', linewidth=2, label=f'Lepton Model (S8={s8_pred:.3f})')
plt.xlabel('Redshift z')
plt.ylabel('Growth Factor D(z)')
plt.title(f'Lepton Model (eta={ETA_FLOOR}) Structure Growth')
plt.legend()
plt.gca().invert_xaxis()
plt.grid(alpha=0.3)
plt.show()
