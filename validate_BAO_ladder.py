# Uncomment the line below if running in Google Colab / Jupyter
# !pip install scipy numpy matplotlib

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

print("==========================================================")
print("   BAO CONSISTENCY AUDIT: GEOMETRIC SCALING (PURE THEORY)")
print("==========================================================")

# ==========================================
# 1. CONSENSUS BAO DATA
# ==========================================
bao_data = [
    # z      val      err      type      Survey
    (0.106,  0.336,   0.015,  'rs_DV',  '6dFGS'),
    (0.15,   4.465,   0.168,  'DV_rs',  'SDSS MGS'),
    (0.38,   1512.0,  25.0,   'DM',     'BOSS DR12'), 
    (0.38,   81.2,    2.4,    'H',      'BOSS DR12'), 
    (0.51,   1975.0,  30.0,   'DM',     'BOSS DR12'),
    (0.51,   90.9,    2.3,    'H',      'BOSS DR12'),
    (0.61,   2307.0,  37.0,   'DM',     'BOSS DR12'),
    (0.61,   97.3,    2.1,    'H',      'BOSS DR12')
]

# ==========================================
# 2. PHYSICS ENGINE (Zero-Parameter Theory)
# ==========================================
C_LIGHT = 299792.458
RD_FIDUCIAL = 147.78
Z_TRANS = 0.641       # EXACT: Topological percolation redshift
WIDTH = 0.10

# --- MODEL A: EXACT Planck 2018 (Baseline) ---
H0_PLANCK = 67.36     # UPDATED: Exact High-Precision Baseline
OM_PLANCK = 0.3153    # UPDATED: Exact High-Precision Baseline
RD_PLANCK = 147.09

# --- MODEL B: Vacuum Elastodynamics (Metric Scaling) ---
H_FAST = 74.69         # UPDATED: Exact High-Precision Geometric Ceiling
H_TERMINAL = 72.71     # UPDATED: Exact High-Precision Terminal Velocity
OM_PRIMORDIAL = 0.3116 # EXACT: Topological Bare Mass
OM_EFFECTIVE = 0.3639  # UPDATED: Exact High-Precision Viscous Load

# EXACT GEOMETRIC CONTRACTION OF THE SOUND HORIZON:
# The BAO ruler is established in the early universe (z > 1000)
# Therefore, it scales strictly with H_FAST.
RD_VAC = RD_PLANCK * (H0_PLANCK / H_FAST) 

# ==========================================
# 3. CALCULATION CORE
# ==========================================
def h_lcdm(z):
    return H0_PLANCK * np.sqrt(OM_PLANCK * (1 + z)**3 + (1 - OM_PLANCK))

def h_viscous_bao(z):
    # DUAL TRANSITION: Density AND Expansion Rate
    # Because BAO observations are at low redshift (z < 1.0),
    # the light travels through the viscous phase transition.
    arg = (Z_TRANS - z) / WIDTH
    sigmoid = np.where(arg > 100, 1.0, np.where(arg < -100, 0.0, 1.0 / (1.0 + np.exp(-arg))))
    
    OM_Z = OM_PRIMORDIAL + (OM_EFFECTIVE - OM_PRIMORDIAL) * sigmoid
    OL_Z = 1.0 - OM_Z
    
    # Kinematic Braking: H_fast (early) -> H_terminal (late)
    H_Z = H_FAST + (H_TERMINAL - H_FAST) * sigmoid
    
    return H_Z * np.sqrt(OM_Z * (1 + z)**3 + OL_Z)

def compute_planck(z):
    Hz = h_lcdm(z)
    dm, _ = quad(lambda z_: C_LIGHT / h_lcdm(z_), 0, z)
    dv = (z * (dm**2) * (C_LIGHT / Hz))**(1.0/3.0)
    return {'H': Hz, 'DM': dm, 'DV': dv}

def compute_vacuum(z):
    Hz = h_viscous_bao(z)
    dm, _ = quad(lambda z_: C_LIGHT / h_viscous_bao(z_), 0, z)
    dv = (z * (dm**2) * (C_LIGHT / Hz))**(1.0/3.0)
    return {'H': Hz, 'DM': dm, 'DV': dv}

# ==========================================
# PART I: THE STATISTICAL ENGINE (8-POINT AUDIT)
# ==========================================
print(f"\n[PART I: 8-POINT CONSENSUS AUDIT]")
print(f"{'Z':<6} | {'Type':<6} | {'Data':<10} | {'Planck':<10} | {'Vacuum':<10} | {'Sigma':<6}")
print("-" * 65)

chi2_planck = 0.0
chi2_vacuum = 0.0

plot_data_z = []
plot_data_val = []
plot_data_err = []

for z, val, err, dtype, survey in bao_data:
    vec_p = compute_planck(z)
    vec_v = compute_vacuum(z) 
    
    if dtype == 'rs_DV':
        pred_p = RD_PLANCK / vec_p['DV']
        pred_v = RD_VAC / vec_v['DV']
    elif dtype == 'DV_rs':
        pred_p = vec_p['DV'] / RD_PLANCK
        pred_v = vec_v['DV'] / RD_VAC
    elif dtype == 'DM':
        val_ratio = val / RD_FIDUCIAL 
        err_ratio = err / RD_FIDUCIAL
        pred_p_val = vec_p['DM'] / RD_PLANCK
        pred_v_val = vec_v['DM'] / RD_VAC
        
        plot_data_z.append(z)
        plot_data_val.append(val_ratio)
        plot_data_err.append(err_ratio)
        
        val, err = val_ratio, err_ratio
        pred_p, pred_v = pred_p_val, pred_v_val
    elif dtype == 'H':
        val_prod = val * RD_FIDUCIAL
        err_prod = err * RD_FIDUCIAL
        pred_p_val = vec_p['H'] * RD_PLANCK
        pred_v_val = vec_v['H'] * RD_VAC
        val, err = val_prod, err_prod
        pred_p, pred_v = pred_p_val, pred_v_val

    chi2_p = ((val - pred_p) / err)**2
    chi2_v = ((val - pred_v) / err)**2
    chi2_planck += chi2_p
    chi2_vacuum += chi2_v
    sigma = (pred_v - val) / err
    print(f"{z:<6.2f} | {dtype:<6} | {val:<10.4f} | {pred_p:<10.4f} | {pred_v:<10.4f} | {sigma:<+6.2f}")

print("-" * 65)
print(f"TOTAL CHI-SQUARED:")
print(f"Planck (Baseline): {chi2_planck:.2f}")
print(f"Vacuum (Theory):   {chi2_vacuum:.2f}")

# ==========================================
# PART II: THE VISUALIZER (BOSS DR12 DM PLOT)
# ==========================================
print("\n[PART II: GENERATING VISUALIZATION...]")

z_grid = np.linspace(0.1, 0.7, 100)
ratio_std_list = []
ratio_vac_list = []

for z_plot in z_grid:
    vec_p = compute_planck(z_plot)
    vec_v = compute_vacuum(z_plot)
    ratio_std_list.append(vec_p['DM'] / RD_PLANCK)
    ratio_vac_list.append(vec_v['DM'] / RD_VAC)

plt.figure(figsize=(10, 6))

plt.errorbar(plot_data_z, plot_data_val, yerr=plot_data_err, fmt='o', color='black', 
             label='BOSS DR12 Data ($D_M / r_d$)', capsize=5, zorder=5)

plt.plot(z_grid, ratio_std_list, 'k--', linewidth=2, label=rf'Standard $\Lambda$CDM ($H_0={H0_PLANCK}$)')
plt.plot(z_grid, ratio_vac_list, 'r-', linewidth=3, 
         label=rf'Theoretical Model ($H_{{fast}} \rightarrow H_{{local}}$)' + '\n' + rf'w/ Contraction ($r_d={RD_VAC:.1f}$ Mpc)')

plt.title('BAO Consistency Check: Geometric Scaling Cancellation', fontsize=14)
plt.xlabel('Redshift $z$', fontsize=12)
plt.ylabel(rf'Transverse BAO Distance $D_M(z) / r_d$', fontsize=12)
plt.legend(loc='lower right')
plt.grid(True, alpha=0.3)

plt.annotate(rf"Global BAO $\chi^2$:" + "\n" + rf"Planck: {chi2_planck:.2f}" + "\n" + rf"Vacuum: {chi2_vacuum:.2f}", 
             xy=(0.05, 0.8), xycoords='axes fraction',
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.8),
             fontsize=11)

plt.tight_layout()
plt.savefig('Figure_BAO_Unified_Audit.pdf', dpi=300)
print("Plot saved to 'Figure_BAO_Unified_Audit.pdf'")
plt.show()
