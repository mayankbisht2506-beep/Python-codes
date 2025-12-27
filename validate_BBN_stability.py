import numpy as np
import matplotlib.pyplot as plt

print("--- HELIUM-4 BBN STABILITY TEST ---")
print("Objective: Verify the 'Cancellation Theorem' (Section 7.9)")

# ==========================================
# 1. PHYSICS CONSTANTS (Standard Model)
# ==========================================
# Fundamental Constants (MeV, seconds)
Q_0 = 1.293          # Neutron-Proton mass difference (MeV)
T_freeze_0 = 0.8     # Freeze-out Temperature (MeV)
TAU_NEUTRON_0 = 880.0 # Free Neutron Lifetime (s)
T_NUC_0 = 180.0      # Time to Deuterium Bottleneck (s)

# Observational Constraint
Yp_OBS = 0.245       # Planck 2018 Helium Fraction
Yp_ERR = 0.003       # Error margin

# ==========================================
# 2. VACUUM ELASTODYNAMICS PARAMETERS
# ==========================================
# Derived in Section 7.1 (Eq. 72)
G_BOOST = 1.23       # Early Gravity (G_early / G_0)

# Geometric Scaling Laws (Section 7.9)
# 1. Mass Scales: m ~ G^-0.5  (Eq. 82)
MASS_FACTOR = G_BOOST**(-0.5)

# 2. Weak Force: G_F ~ 1/v^2 ~ 1/m^2 ~ G^1.0 (Eq. 94 text)
GF_FACTOR = G_BOOST**(1.0) 

# 3. Expansion Rate: H ~ sqrt(G) * T^2 (Eq. 94)
H_FACTOR = np.sqrt(G_BOOST)

def calculate_helium_fraction(model='std'):
    """
    Calculates Primordial Helium (Yp) based on vacuum parameters.
    """
    if model == 'std':
        # Standard LambdaCDM Parameters
        Q = Q_0
        G_F_scale = 1.0
        H_scale = 1.0
        tau_n = TAU_NEUTRON_0
        t_nuc = T_NUC_0
    else:
        # Vacuum Elastodynamics Parameters
        # 1. Mass Difference scales with Mass (Q ~ m ~ G^-0.5)
        Q = Q_0 * MASS_FACTOR
        
        # 2. Weak Coupling (G_F ~ G)
        G_F_scale = GF_FACTOR
        
        # 3. Expansion Rate (H ~ G^0.5)
        H_scale = H_FACTOR
        
        # 4. Neutron Lifetime (Eq. 96)
        # Gamma_decay ~ G_F^2 * m_e^5 ~ (G)^2 * (G^-0.5)^5 ~ G^-0.5
        # Lifetime (tau) ~ 1/Gamma ~ G^0.5
        tau_n = TAU_NEUTRON_0 * (G_BOOST**0.5)
        
        # 5. Nucleosynthesis Time (Eq. 97)
        # Target Temp depends on Binding Energy (B_D ~ m ~ G^-0.5)
        # Time t ~ 1/(sqrt(G)*T^2) -> scales as G^0.5
        t_nuc = T_NUC_0 * (G_BOOST**0.5)

    # --- STEP A: Freeze-Out Temperature ---
    # Freeze-out: Gamma_weak ~ H
    # G_F^2 * T^5 ~ H * T^2  ->  T^3 ~ H / G_F^2
    T_freeze = T_freeze_0 * (H_scale / (G_F_scale**2))**(1.0/3.0)
    
    # --- STEP B: Neutron-to-Proton Ratio ---
    # n/p = exp(-Q / T_freeze)
    # Note: In Vacuum model, Q drops and T drops, cancelling out [cite: 939]
    np_ratio_freeze = np.exp(-Q / T_freeze)
    
    # --- STEP C: Neutron Decay ---
    # Fraction surviving until nucleosynthesis
    # decay_exponent = t_nuc / tau_n
    # Note: In Vacuum model, both t_nuc and tau_n increase by G^0.5 [cite: 955]
    decay_fraction = np.exp(-t_nuc / tau_n)
    
    np_ratio_final = np_ratio_freeze * decay_fraction
    
    # --- STEP D: Helium Yield ---
    Yp = 2 * np_ratio_final / (1 + np_ratio_final)
    
    return Yp, T_freeze, Q, t_nuc

# ==========================================
# 3. EXECUTE TEST
# ==========================================
Yp_std, Tf_std, Q_std, tn_std = calculate_helium_fraction('std')
Yp_vac, Tf_vac, Q_vac, tn_vac = calculate_helium_fraction('vac')

diff = Yp_vac - Yp_std
percent_diff = (diff / Yp_std) * 100

print("\n" + "="*60)
print("RESULTS: PRIMORDIAL HELIUM ABUNDANCE (Yp)")
print("="*60)
print(f"{'Parameter':<20} | {'Standard Model':<15} | {'Vacuum Model':<15} | {'Scaling'}")
print("-" * 60)
print(f"{'Q (Mass Diff)':<20} | {Q_std:<15.4f} | {Q_vac:<15.4f} | G^-0.5")
print(f"{'T_freeze (MeV)':<20} | {Tf_std:<15.4f} | {Tf_vac:<15.4f} | G^-0.5")
print(f"{'Time to BBN (s)':<20} | {tn_std:<15.1f} | {tn_vac:<15.1f} | G^0.5")
print("-" * 60)
print(f"{'Final Yp':<20} | {Yp_std:<15.5f} | {Yp_vac:<15.5f} | INVARIANT")
print("="*60)
print(f"Difference: {diff:.6f} ({percent_diff:.3f}%)")

if abs(percent_diff) < 0.1:
    print("VERDICT: PERFECT CANCELLATION (Pass).")
    print("Matches Section 7.9 proof: Nuclear rates scale identically to expansion.")
else:
    print("VERDICT: FAILURE. Check scaling laws.")

# ==========================================
# 4. PLOTTING
# ==========================================
plt.figure(figsize=(8, 6))
x = ['Standard $\Lambda$CDM', 'Vacuum Elastodynamics']
y = [Yp_std, Yp_vac]
colors = ['gray', '#2ca02c'] # Green for Vacuum

bars = plt.bar(x, y, color=colors, width=0.6)
plt.axhline(Yp_OBS, color='blue', linestyle='--', label=f'Planck Obs ({Yp_OBS})')
plt.axhspan(Yp_OBS - Yp_ERR, Yp_OBS + Yp_ERR, color='blue', alpha=0.1)

# Annotations
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.002,
             f'{height:.4f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.ylim(0.20, 0.30)
plt.ylabel('Primordial Helium Fraction ($Y_p$)', fontsize=12)
plt.title('BBN Invariance Test: The Cancellation Theorem', fontsize=14)
plt.legend(loc='lower right')
plt.grid(axis='y', linestyle=':', alpha=0.5)

plt.tight_layout()
plt.savefig('validate_BBN_stability.png')
print("Plot saved as 'validate_BBN_stability.png'")
plt.show()
