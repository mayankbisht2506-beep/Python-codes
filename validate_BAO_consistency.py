import numpy as np
import matplotlib.pyplot as plt

print("--- BAO CHI-SQUARED TEST: FINAL VALIDATION ---")

# ==========================================
# 1. DATA (Consensus "Gold" Dataset)
# ==========================================
bao_data = [
    {'z': 0.106, 'val': 0.336,   'err': 0.015,  'type': 'rs_DV'}, # 6dFGS
    {'z': 0.15,  'val': 4.466,   'err': 0.168,  'type': 'DV_rs'}, # SDSS MGS
    {'z': 0.38,  'val': 1512.39, 'err': 25.0,   'type': 'DM'},    # BOSS DR12
    {'z': 0.38,  'val': 81.208,  'err': 2.4,    'type': 'H'},     # BOSS DR12
    {'z': 0.61,  'val': 2306.68, 'err': 37.0,   'type': 'DM'},    # BOSS DR12
    {'z': 0.61,  'val': 97.26,   'err': 2.1,    'type': 'H'}      # BOSS DR12
]

# ==========================================
# 2. PHYSICS CONSTANTS
# ==========================================
C_LIGHT = 299792.458
RD_FIDUCIAL = 147.78 

# --- Model A: Planck 2018 ---
H0_PLANCK = 67.4
OM_PLANCK = 0.315
RD_PLANCK = 147.09

# --- Model B: Vacuum Elastodynamics ---
H0_VAC = 74.5
OM_VAC = 0.350  # Updated Matter Density
RD_VAC = 147.09 * 0.905  # Superfluid Contraction (9.5%)
Z_TRANS = 0.65
WIDTH = 0.10

# ==========================================
# 3. HELPER FUNCTIONS
# ==========================================
def get_h_lcdm(z, h0, om):
    ol = 1.0 - om
    return h0 * np.sqrt(om * (1 + z)**3 + ol)

def get_h_vacuum(z, h0_local, h0_early, om, z_trans, width):
    # Base expansion using Vacuum Density (OM_VAC)
    # Note: h0_early is used for the base scale
    h_base = get_h_lcdm(z, h0_early, om)
    
    # Stiffness Transition Boost
    sigmoid = 1.0 / (1.0 + np.exp((z - z_trans) / width))
    boost = 1.0 + ((h0_local/h0_early) - 1.0) * sigmoid
    
    return h_base * boost

def compute_vectors(z_target, h_func):
    # Integration grid
    z_grid = np.linspace(0, z_target, 1000)
    h_vals = h_func(z_grid)
    
    # Comoving Distance (Trapezoidal Rule)
    integrand = C_LIGHT / h_vals
    dm = np.sum((integrand[:-1] + integrand[1:]) / 2 * np.diff(z_grid))

    # Hubble and DV
    h_val = h_func(z_target) # Scalar at target
    dv = (z_target * (C_LIGHT/h_val) * dm**2)**(1.0/3.0)
    
    return {'DM': dm, 'H': h_val, 'DV': dv}

def get_prediction(pt, vec, rd_model):
    # Scales prediction to match Fiducial data units
    scale = RD_FIDUCIAL / rd_model
    
    if pt['type'] == 'rs_DV': return rd_model / vec['DV']
    if pt['type'] == 'DV_rs': return vec['DV'] / rd_model
    if pt['type'] == 'DM':    return vec['DM'] * scale
    if pt['type'] == 'H':     return vec['H'] / scale
    return 0.0

# ==========================================
# 4. RUN TEST
# ==========================================
print(f"{'Z':<4} | {'Data':<9} | {'Planck':<9} | {'Vacuum':<9}")
print("-" * 50)

chi2_planck = 0.0
chi2_vacuum = 0.0

for pt in bao_data:
    z = pt['z']
    
    # 1. Compute Planck Vector
    # Uses H0=67.4, OM=0.315
    h_func_p = lambda z_: get_h_lcdm(z_, H0_PLANCK, OM_PLANCK)
    vec_p = compute_vectors(z, h_func_p)
    pred_p = get_prediction(pt, vec_p, RD_PLANCK)
    
    # 2. Compute Vacuum Vector
    # Uses H0=74.5, OM=0.350, Transition at 0.65
    h_func_v = lambda z_: get_h_vacuum(z_, H0_VAC, H0_PLANCK, OM_VAC, Z_TRANS, WIDTH)
    vec_v = compute_vectors(z, h_func_v)
    pred_v = get_prediction(pt, vec_v, RD_VAC)
    
    # 3. Accumulate Chi2
    chi2_planck += ((pt['val'] - pred_p) / pt['err'])**2
    chi2_vacuum += ((pt['val'] - pred_v) / pt['err'])**2
    
    print(f"{z:<4.2f} | {pt['val']:<9.2f} | {pred_p:<9.2f} | {pred_v:<9.2f}")

print("-" * 50)
print(f"Chi2 Planck: {chi2_planck:.2f}")
print(f"Chi2 Vacuum: {chi2_vacuum:.2f}")

if chi2_vacuum < chi2_planck + 5.0:
    print("VERDICT: SUCCESS. Vacuum Model is statistically consistent.")
else:
    print("VERDICT: TENSION. Check parameters.")
