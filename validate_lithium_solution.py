# Uncomment the line below if running in Google Colab / Jupyter
# !pip install numpy scipy

import numpy as np
from scipy.integrate import trapezoid

def verify_lithium_depletion():
    print("----------------------------------------------------------------")
    print("   VACUUM ELASTODYNAMICS: LITHIUM-7 VERIFICATION SUITE")
    print("   Mechanism: A=8 Interstitial Lattice Softening (Bag Inflation)")
    print("----------------------------------------------------------------")

    # ==========================================
    # 1. GEOMETRIC INPUT PARAMETERS (Exact Topology)
    # ==========================================
    CABIBBO_ANGLE = 0.225   # Standard Model Mixing Angle (sin theta_c)
    Y_MAX = 0.2090          # EXACT: Updated Macroscopic Yield Limit (beta_p = 0.418)

    # Derived Effective Stiffness and Gravity Boost
    DELTA_EFF = CABIBBO_ANGLE * (1.0 - Y_MAX)
    G_BOOST = 1.0 / (1.0 - DELTA_EFF)

    # ==========================================
    # 2. THE VED TOPOLOGICAL SYMMETRY BREAKING
    # ==========================================
    # As proven in Sec 7.11.1 & 7.11.2, the universal background scalings (H, T, bulk mass)
    # perfectly cancel to G^0 = 1, protecting Deuterium and Helium-4.
    
    # However, for A=8 (Lithium + proton -> 8Be*), dense packing forces interstitial softening.
    # The secondary effective mass compliance ratio is m_eff(A=8) / m(G) = G^-0.5
    A8_MASS_COMPLIANCE = G_BOOST**(-0.5)

    # The Coulomb barrier scales with the square root of this uncompensated mass deficit:
    A8_BARRIER_RATIO = np.sqrt(A8_MASS_COMPLIANCE) # equals G^-0.25 (approx 0.952)

    print("Physics Parameters:")
    print(f"  > Gravity (G_early):           {G_BOOST:.4f} x G0")
    print(f"  > Bulk Plasma Scaling:         G^0 = 1.000 (D & He4 Protected)")
    print(f"  > A=8 Secondary Mass Deficit:  {A8_MASS_COMPLIANCE:.4f}x")
    print(f"  > A=8 Coulomb Barrier Ratio:   {A8_BARRIER_RATIO:.4f}x (Weaker Barrier)")
    print("-" * 64)

    # ==========================================
    # 3. PHYSICS ENGINE (Gamow Integration)
    # ==========================================
    # Scaled Caughlan-Fowler barrier constant for Li7(p,alpha)He4 reaction
    B0 = 84.72              

    def calculate_burn_exponent(model='std'):
        """
        Integrates the reaction rate exponent over the scaled cooling history.
        The Survival Fraction S = exp(- Integral(Rate * dt))
        """
        # Using the rescaled dimensionless BBN temperature frame (T_tilde)
        T_tilde = np.logspace(np.log10(3.0), np.log10(0.1), 10000)
        
        if model == 'vac':
            # ONLY the A=8 uncompensated barrier softening is applied.
            # All other global time/density terms cancel to 1.0.
            barrier_multiplier = A8_BARRIER_RATIO
        else:
            barrier_multiplier = 1.0

        # 1. Reaction Rate (Proportional Gamow Tunneling)
        B_effective = B0 * barrier_multiplier
        tau = B_effective / (T_tilde**(1.0/3.0))
        rate_kernel = (T_tilde**(-2.0/3.0)) * np.exp(-tau)

        # 2. Time Measure (dt)
        dt_proportional = (T_tilde**-3) 
        
        # 3. Integrate (Rate * dt)
        integrand = rate_kernel * dt_proportional
        total_burn = trapezoid(integrand, x=T_tilde) 
        
        return np.abs(total_burn)

    # ==========================================
    # 4. EXECUTION & CALIBRATION
    # ==========================================
    integral_std = calculate_burn_exponent('std')
    integral_vac = calculate_burn_exponent('vac')

    # Calibrate to Standard Model Reference
    REF_SURVIVAL_STD = 0.938
    calibration_C = -np.log(REF_SURVIVAL_STD) / integral_std

    # Calculate Vacuum Prediction
    survival_vac = np.exp(-calibration_C * integral_vac)
    depletion_factor = REF_SURVIVAL_STD / survival_vac

    # ==========================================
    # 5. RESULTS REPORTING
    # ==========================================
    print("RESULTS:")
    print(f"  > Standard Model Survival: {REF_SURVIVAL_STD*100:.2f}% (Reference)")
    print(f"  > Vacuum Model Survival:   {survival_vac*100:.2f}%")
    print(f"  > Depletion Factor:        {depletion_factor:.2f}x")
    print("-" * 64)

    if 2.5 <= depletion_factor <= 3.5:
        print("[SUCCESS] VERDICT: PASS.")
        print(f"The model successfully predicts the ~{depletion_factor:.2f}x cosmological depletion factor.")
        print("Mechanism: Secondary Interstitial Softening selectively breaks the")
        print("G^0 adiabatic invariance strictly for the A=8 compound state.")
    else:
        print(f"[WARNING] VERDICT: DEVIATION. Factor {depletion_factor:.2f}x")

if __name__ == "__main__":
    verify_lithium_depletion()
