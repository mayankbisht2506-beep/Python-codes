import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. BAO DATA (Consensus "Gold" Set)
# ==========================================
# Format: z_eff, value, error, type
# CORRECTED: 6dFGS is DV/rs, not rs/DV
bao_data = [
    # 6dFGS (Beutler 2011) - DV/rd = 2.976
    {'z': 0.106, 'val': 2.976, 'err': 0.133, 'type': 'DV_rs'},
    # SDSS MGS (Ross 2015) - DV/rd = 4.466
    {'z': 0.15,  'val': 4.466, 'err': 0.168, 'type': 'DV_rs'},
    # BOSS DR12 (Alam 2017) - Low z
    {'z': 0.38,  'val': 1512.39, 'err': 25.0, 'type': 'DM'}, # Mpc
    {'z': 0.38,  'val': 81.208,  'err': 2.4,  'type': 'H'},  # km/s/Mpc
    # BOSS DR12 - High z
    {'z': 0.61,  'val': 2306.68, 'err': 37.0, 'type': 'DM'},
    {'z': 0.61,  'val': 97.26,   'err': 2.1,  'type': 'H'}
]

# Constants
C_LIGHT = 299792.458
RD_PLANCK = 147.09  # Fixed Anchor
H0_PLANCK = 67.4
OM_PLANCK = 0.315
H0_SHOES = 73.04

# ==========================================
# 2. PHYSICS ENGINE
# ==========================================
Z_TRANS = 0.65
WIDTH = 0.1

def get_h_viscous(z):
    # Viscous Transition H(z)
    sigmoid = 1 / (1 + np.exp((z - Z_TRANS) / WIDTH))
    h0_eff = H0_PLANCK + (H0_SHOES - H0_PLANCK) * sigmoid
    E_z = np.sqrt(OM_PLANCK * (1 + z)**3 + (1 - OM_PLANCK))
    return h0_eff * E_z

def get_h_planck(z):
    # Standard LCDM
    return H0_PLANCK * np.sqrt(OM_PLANCK * (1 + z)**3 + (1 - OM_PLANCK))

def compute_distances(z_target, h_func):
    """Integrates H(z) to get geometric distances"""
    z_grid = np.linspace(0, z_target, 1000)
    h_grid = h_func(z_grid)
    integrand = C_LIGHT / h_grid

    # Comoving Distance D_M
    dm = np.trapz(integrand, z_grid)

    # Hubble Distance D_H
    dh = C_LIGHT / h_func(z_target)

    # Spherically Averaged D_V
    dv = (z_target * dh * dm**2)**(1.0/3.0)

    return {'DM': dm, 'DH': dh, 'DV': dv, 'H': h_func(z_target)}

# ==========================================
# 3. RUN THE TEST
# ==========================================
chi2_planck = 0
chi2_viscous = 0

print(f"{'Dataset':<10} | {'z':<5} | {'Observed':<10} | {'Planck':<10} | {'Viscous':<10} | {'Sigma(Visc)':<5}")
print("-" * 75)

for pt in bao_data:
    z = pt['z']
    obs = pt['val']
    err = pt['err']

    # Compute Models
    d_planck = compute_distances(z, get_h_planck)
    d_visc = compute_distances(z, get_h_viscous)

    # Match Data Type
    if pt['type'] == 'rs_DV':
        pred_p = RD_PLANCK / d_planck['DV']
        pred_v = RD_PLANCK / d_visc['DV']
    elif pt['type'] == 'DV_rs':
        pred_p = d_planck['DV'] / RD_PLANCK
        pred_v = d_visc['DV'] / RD_PLANCK
    elif pt['type'] == 'DM':
        pred_p = d_planck['DM']
        pred_v = d_visc['DM']
    elif pt['type'] == 'H':
        pred_p = d_planck['H']
        pred_v = d_visc['H']

    # Chi2 Contribution
    c2_p = ((obs - pred_p) / err)**2
    c2_v = ((obs - pred_v) / err)**2

    chi2_planck += c2_p
    chi2_viscous += c2_v

    # Sigma Pull
    sigma = (pred_v - obs) / err

    print(f"{pt.get('type'):<10} | {z:<5.2f} | {obs:<10.2f} | {pred_p:<10.2f} | {pred_v:<10.2f} | {sigma:<+5.2f}")

print("-" * 75)
print(f"TOTAL CHI2 (Planck Baseline):   {chi2_planck:.2f}")
print(f"TOTAL CHI2 (Vacuum Viscosity):  {chi2_viscous:.2f}")
print(f"Delta Chi2 (BAO):               {chi2_viscous - chi2_planck:.2f}")

# Plot
z_plot = np.linspace(0.01, 0.8, 100)
ratios_visc = []
for z in z_plot:
    d_p = compute_distances(z, get_h_planck)
    d_v = compute_distances(z, get_h_viscous)
    ratios_visc.append(d_v['DV'] / d_p['DV'])

plt.figure(figsize=(8,5))
plt.plot(z_plot, ratios_visc, 'r-', linewidth=3, label='Viscous Model')
plt.axhline(1.0, color='k', linestyle='--', label='Planck Baseline')
plt.xlabel('Redshift z')
plt.ylabel(r'$D_V(z)$ Ratio (Model / Planck)')
plt.title(r'BAO Consistency Check ($D_V$ Contraction)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('Figure_BAO_Check.png')
print("Plot saved.")
