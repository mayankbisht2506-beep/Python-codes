import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# ==========================================
# 1. PHYSICS INPUTS & GEOMETRIC SCALINGS
# ==========================================
B0 = 84.0  # Gamow Constant for Li7+p

# Vacuum Parameters (From Section 7.1 & 9.6)
G_BOOST = 1.23  # G_early / G_0

# A. Mass Scaling (Tunneling Barrier)
# Nucleons 10% lighter -> Lower Coulomb Barrier -> Exponential Rate Boost
# m ~ G^-0.5
MASS_SCALE_VAC = G_BOOST**(-0.5)  # 0.9015

# B. Time Scaling (Integration Window)
# Universe expands slower -> More time for burning
# t ~ G^0.5
TIME_SCALE_VAC = G_BOOST**(0.5)   # 1.109

# C. Cross-Section Scaling (Geometric Size)
# "Fluffier" nucleons -> Larger target area -> Linear Rate Boost
# sigma ~ G^1.0 (Section 7.9.3, Eq. 100)
SIGMA_SCALE_VAC = G_BOOST**(1.0)  # 1.23

# ==========================================
# 2. CALIBRATED REACTION RATE
# ==========================================
# Calibrated to ensure standard model roughly matches standard theory survival (~94%)
RATE_CONST = 2.5e36

def reaction_rate(T_GK, mass_scale=1.0):
    if T_GK <= 0.05: return 0.0
    # Tunneling exponent depends on reduced mass (Gamow Window)
    # B_eff ~ m^(1/3)
    B_eff = B0 * (mass_scale)**(1.0/3.0)
    tau = B_eff / (T_GK**(1.0/3.0))
    return (T_GK**(-2.0/3.0)) * np.exp(-tau)

def depletion_ode(y, t, model='std'):
    Y = y[0]
    
    # Temperature evolution T ~ 1/sqrt(t)
    T = 1.0 / np.sqrt(t)

    if model == 'std':
        m_scale = 1.0
        sigma_boost = 1.0
    else:
        # Vacuum Model Physics
        m_scale = MASS_SCALE_VAC   # Lowers the Barrier (Exponential boost)
        sigma_boost = SIGMA_SCALE_VAC # Increases Target Size (Linear boost)

    # Calculate base rate based on Temperature and Mass
    raw_rate = reaction_rate(T, m_scale)
    
    # Apply total scaling
    # Rate = Constant * (Tunneling Physics) * (Geometric Cross-Section)
    total_rate = RATE_CONST * raw_rate * sigma_boost
    
    return -total_rate * Y

# ==========================================
# 3. RUN SIMULATION
# ==========================================
t_start = 1.0
t_end_std = 100.0
# Vacuum model integrates longer due to time dilation (t ~ G^0.5)
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
print(f"Physics: Mass Scaling (0.90x) + Sigma Boost (1.23x) + Time Dilation (1.11x)")
print(f"Standard Model Survival: {final_std*100:.1f}%")
print(f"Vacuum Model Survival:   {final_vac*100:.1f}%")
print(f"Resolution Factor:       {resolution_factor:.2f}x (Target ~3.0)")

# ==========================================
# 5. PLOT
# ==========================================
plt.figure(figsize=(10,6))
# Map time back to Temperature for plotting
T_axis_std = 1.0/np.sqrt(t_std)
# For vac, T is comparable at equivalent evolutionary phases
T_axis_vac = 1.0/np.sqrt(np.linspace(t_start, t_end_std, 1000)) 
# Note: Plotting against T aligns the cooling tracks visually

plt.plot(T_axis_std, sol_std, 'k--', linewidth=2, label=r'Standard $\Lambda$CDM')
plt.plot(T_axis_std, sol_vac, 'r-', linewidth=3, label='Vacuum Elastodynamics') # Plot against same T axis to show depletion depth

plt.xlim(0.8, 0.2)
plt.xlabel('Temperature (GK)')
plt.ylabel('Lithium-7 Abundance (Normalized)')
plt.title(f'Resolution of Lithium Anomaly (Factor {resolution_factor:.1f}x)')
plt.grid(True, alpha=0.3)
plt.legend(loc='lower left')

plt.savefig('Figure_Li7_Final.png')
print("Plot saved as Figure_Li7_Final.png")
plt.show()

