import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. PHYSICS SETUP
# ==========================================
# Constants
Q_0 = 1.293   # Neutron-Proton mass diff (MeV) in Standard Model
T_freeze_0 = 0.8 # Standard Freeze-out Temp (MeV)
Yp_obs = 0.245   # Observed Helium Fraction (+/- 0.003)

# Vacuum Parameters (from Paper)
G_BOOST = 1.23
# Mass Scaling: m ~ G^-0.5
MASS_FACTOR = G_BOOST**(-0.5) # 0.9015
# Weak Force Scaling: G_F ~ 1/v^2 ~ 1/m^2 ~ G
GF_FACTOR = G_BOOST**(1.0)    # 1.23 (Stronger Weak Force)

# ==========================================
# 2. CALCULATION
# ==========================================
def calculate_helium(model='std'):
    if model == 'std':
        H_scale = 1.0
        Q = Q_0
        GF = 1.0
    else:
        # H ~ sqrt(G)
        H_scale = np.sqrt(G_BOOST)
        # Mass diff scales with mass
        Q = Q_0 * MASS_FACTOR
        GF = GF_FACTOR

    # WEAK RATE: Gamma ~ G_F^2 * T^5
    # EXPANSION: H ~ sqrt(G) * T^2
    # Freeze-out happens when Gamma ~ H
    # (G_F^2 * T^5) ~ H_scale * T^2
    # T^3 ~ H_scale / G_F^2
    # T_freeze ~ (H_scale / GF^2)^(1/3) * T_freeze_0

    # Calculate Shift in Freeze-out Temperature
    ratio = H_scale / (GF**2)
    T_f = T_freeze_0 * (ratio)**(1.0/3.0)

    # Calculate n/p Ratio at Freeze-out
    # n/p = exp(-Q / T)
    np_ratio_freeze = np.exp(-Q / T_f)

    # Neutron Decay Correction (Free neutrons decay before forming He)
    # Time to Nucleosynthesis t_nuc approx 180s standard
    # In Vacuum: t ~ 1/sqrt(G) -> Faster
    t_nuc = 180.0 / np.sqrt(H_scale) # Simplified scaling
    tau_neutron = 880.0 # Lifetime in seconds

    decay_factor = np.exp(-t_nuc / tau_neutron)
    np_ratio_final = np_ratio_freeze * decay_factor

    # Helium Fraction Yp = 2(n/p) / (1 + n/p)
    Yp = 2 * np_ratio_final / (1 + np_ratio_final)

    return Yp, T_f, Q

# ==========================================
# 3. RUN TEST
# ==========================================
Yp_std, Tf_std, Q_std = calculate_helium('std')
Yp_vac, Tf_vac, Q_vac = calculate_helium('vac')

diff = abs(Yp_vac - Yp_std)
percent_diff = (diff / Yp_std) * 100

print(f"--- HELIUM-4 STABILITY CHECK ---")
print(f"Standard Model Yp: {Yp_std:.4f} (Obs: {Yp_obs})")
print(f"Vacuum Model Yp:   {Yp_vac:.4f}")
print(f"Difference:       {diff:.4f} ({percent_diff:.2f}%)")

if percent_diff < 5.0:
    print("VERDICT: STABLE (Success)")
    print("The stronger Weak Force cancels the faster expansion.")
else:
    print("VERDICT: UNSTABLE (Risk)")

# ==========================================
# 4. PLOT (Comparison)
# ==========================================
labels = ['Standard LCDM', 'Vacuum Elastodynamics']
values = [Yp_std, Yp_vac]

plt.figure(figsize=(8, 5))
bars = plt.bar(labels, values, color=['gray', 'green' if percent_diff < 5 else 'red'])
plt.axhline(Yp_obs, color='blue', linestyle='--', linewidth=2, label='Observation (0.245)')
plt.ylim(0.20, 0.30)
plt.ylabel('Helium-4 Mass Fraction (Yp)')
plt.title(f'Helium-4 Stability Test (Delta = {percent_diff:.1f}%)')
plt.legend()

# Add text on bars
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.002, round(yval, 4), ha='center', va='bottom')

plt.savefig('Figure_He4_Stability.png')
plt.show()
