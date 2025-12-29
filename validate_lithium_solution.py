import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

print("--- LITHIUM-7 BURNING SIMULATION (STRICT) ---")

# ==========================================
# 1. PHYSICS INPUTS (STRICT ADD 33)
# ==========================================
B0 = 84.0  # Gamow Constant for Li7+p

# Grand Unification Parameters
H0_PLANCK = 67.4
H0_THEORY = 74.5   # The Gravity Boost Prediction

# CALCULATE G_BOOST EXACTLY
# G ~ H^2 (Friedmann Eq dominant term)
G_BOOST = (H0_THEORY / H0_PLANCK)**2  # approx 1.2216

# A. Mass Scaling (Tunneling Barrier)
# m ~ G^-0.5
MASS_SCALE_VAC = G_BOOST**(-0.5)

# B. Time Scaling (Integration Window)
# t ~ G^0.5
TIME_SCALE_VAC = G_BOOST**(0.5)

# C. Cross-Section Scaling (Geometric Size)
# sigma ~ G^1.0
SIGMA_SCALE_VAC = G_BOOST**(1.0)

# ==========================================
# 2. CALIBRATED REACTION RATE
# ==========================================
RATE_CONST = 2.5e36

def reaction_rate(T_GK, mass_scale=1.0):
    if T_GK <= 0.05: return 0.0
    # Tunneling: B_eff ~ m^(1/3)
    B_eff = B0 * (mass_scale)**(1.0/3.0)
    tau = B_eff / (T_GK**(1.0/3.0))
    return (T_GK**(-2.0/3.0)) * np.exp(-tau)

def depletion_ode(y, t, model='std'):
    Y = y[0]
    T = 1.0 / np.sqrt(t)

    if model == 'std':
        m_scale = 1.0
        sigma_boost = 1.0
    else:
        m_scale = MASS_SCALE_VAC   
        sigma_boost = SIGMA_SCALE_VAC 

    raw_rate = reaction_rate(T, m_scale)
    total_rate = RATE_CONST * raw_rate * sigma_boost
    
    return -total_rate * Y

# ==========================================
# 3. RUN SIMULATION
# ==========================================
t_start = 1.0
t_end_std = 100.0
t_end_vac = 100.0 * TIME_SCALE_VAC

# Standard Model
t_std = np.linspace(t_start, t_end_std, 1000)
sol_std = odeint(depletion_ode, [1.0], t_std, args=('std',))
final_std = sol_std[-1, 0]

# Vacuum Model
t_vac = np.linspace(t_start, t_end_vac, 1000)
sol_vac = odeint(depletion_ode, [1.0], t_vac, args=('vac',))
final_vac = sol_vac[-1, 0]

# ==========================================
# 4. RESULTS
# ==========================================
resolution_factor = final_std / final_vac

print(f"--- LITHIUM-7 FINAL VERIFICATION ---")
print(f"H0 Theory:           {H0_THEORY} (Exact G_BOOST = {G_BOOST:.4f})")
print("-" * 50)
print(f"Physics Scaling:")
print(f"  > Mass (Tunneling): {MASS_SCALE_VAC:.4f}")
print(f"  > Sigma (Target):   {SIGMA_SCALE_VAC:.4f}")
print(f"  > Time (Window):    {TIME_SCALE_VAC:.4f}")
print("-" * 50)
print(f"Standard Survival:   {final_std*100:.2f}%")
print(f"Vacuum Survival:     {final_vac*100:.2f}%")
print(f"Depletion Factor:    {resolution_factor:.2f}x")
print("-" * 50)

if resolution_factor > 2.5:
    print("VERDICT: PASS. Solving the Cosmological Lithium Problem.")
else:
    print(f"VERDICT: PARTIAL. Factor {resolution_factor:.2f}x is helpful but maybe not full solution.")

# Plot
plt.figure(figsize=(9,6))
T_axis_std = 1.0/np.sqrt(t_std)
T_axis_vac = 1.0/np.sqrt(t_vac)

plt.plot(T_axis_std, sol_std, 'k--', linewidth=2, label=r'Standard $\Lambda$CDM')
plt.plot(T_axis_vac, sol_vac, 'r-', linewidth=3, label='Vacuum Elastodynamics') 



plt.xlim(0.8, 0.08) 
plt.xlabel('Temperature ($T_9$)')
plt.ylabel('Lithium-7 Abundance (Normalized)')
plt.title(f'Strict Solution: Lithium Anomaly (Factor {resolution_factor:.1f}x)')
plt.grid(True, alpha=0.3)
plt.legend(loc='lower left')
plt.savefig('Figure_Li7_Strict.png')
print("Plot saved as Figure_Li7_Strict.png")
plt.show()
