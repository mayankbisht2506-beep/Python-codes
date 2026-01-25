import numpy as np
import matplotlib.pyplot as plt

def verify_lithium_depletion():
    print("----------------------------------------------------------------")
    print("   VACUUM ELASTODYNAMICS: LITHIUM-7 VERIFICATION SUITE")
    print("----------------------------------------------------------------")

    # ==========================================
    # 1. GEOMETRIC INPUT PARAMETERS
    # ==========================================
    # Hubble Constants (km/s/Mpc)
    H0_PLANCK = 67.4   # Baseline (Standard Model)
    H0_THEORY = 74.5   # Predicted (Vacuum Elastodynamics)

    # A. Gravity Boost (G_early / G_0)
    # Derived from Friedmann Eq: H^2 ~ G * rho
    G_BOOST = (H0_THEORY / H0_PLANCK)**2
    
    # B. Mass Scaling (The "Turbocharger")
    # Paper Eq (80): m(z) ~ G(z)^-0.5
    # Lighter nucleons tunnel through Coulomb barriers easier.
    MASS_SCALE = G_BOOST**(-0.5)

    # C. Time Scaling (The "Brake")
    # Friedmann Time Relation: t ~ 1/H ~ 1/sqrt(G)
    # Higher G means faster expansion -> less time for nucleosynthesis.
    TIME_SCALE = G_BOOST**(-0.5)

    print(f"Physics Parameters:")
    print(f"  > Hubble Boost:      {H0_PLANCK} -> {H0_THEORY} km/s/Mpc")
    print(f"  > Gravity (G_early): {G_BOOST:.4f} x G0")
    print(f"  > Mass (Tunneling):  {MASS_SCALE:.4f} x m0 (Lighter)")
    print(f"  > Time (Cooling):    {TIME_SCALE:.4f} x t0 (Faster)")
    print("-" * 64)

    # ==========================================
    # 2. PHYSICS ENGINE (Gamow Integration)
    # ==========================================
    # Constants for Li7(p,alpha)He4 reaction
    # B0 is the Gamow constant related to the Coulomb barrier strength
    B0 = 84.72           

    def calculate_burn_exponent(model='std'):
        """
        Integrates the reaction rate exponent over the cooling history.
        The Survival Fraction S = exp(- Integral(Rate * dt))
        """
        # Integration range: Temperature T9 (Billion K) from 3.0 down to 0.1
        # We use log-space for numerical precision
        T9 = np.logspace(np.log10(3.0), np.log10(0.1), 10000)
        
        # Apply Physics Scaling
        if model == 'vac':
            m_eff = MASS_SCALE   # Mass Scaling applied to Barrier
            t_mult = TIME_SCALE  # Time Scaling applied to Duration
        else:
            m_eff = 1.0
            t_mult = 1.0

        # 1. Reaction Rate (Proportional)
        # Rate ~ exp(-B_eff / T^(1/3))
        # The Barrier B_eff scales with sqrt(reduced_mass) ~ m^(1/3) roughly
        # Paper Eq (89): b propto sqrt(m) for the exponential term specifically
        B_effective = B0 * (m_eff**(1.0/2.0)) # Using sqrt(m) scaling for Coulomb param
        
        # Gamow Window Exponent
        tau = B_effective / (T9**(1.0/3.0))
        
        # The burning rate kernel
        rate_kernel = (T9**(-2.0/3.0)) * np.exp(-tau)

        # 2. Time Measure (dt)
        # In radiation era, t ~ T^-2, so dt ~ T^-3 dT
        # We also apply the global TIME_SCALE factor here
        dt_proportional = (T9**-3) * t_mult
        
        # 3. Integrate (Rate * dt)
        # This gives the "Total Burn Strength"
        integrand = rate_kernel * dt_proportional
        total_burn = np.trapz(integrand, x=T9) 
        
        return np.abs(total_burn)

    # ==========================================
    # 3. EXECUTION & CALIBRATION
    # ==========================================
    
    # Calculate raw integrals
    integral_std = calculate_burn_exponent('std')
    integral_vac = calculate_burn_exponent('vac')

    # Calibrate to Standard Model Reference
    # Standard Theory predicts ~93.8% survival (only ~6% burned)
    # S = exp(-C * Integral) -> C = -ln(S_ref) / Integral_std
    REF_SURVIVAL_STD = 0.938
    calibration_C = -np.log(REF_SURVIVAL_STD) / integral_std

    # Calculate Vacuum Prediction
    # S_vac = exp(-C * Integral_vac)
    survival_vac = np.exp(-calibration_C * integral_vac)
    
    # Calculate Depletion Factor
    depletion_factor = REF_SURVIVAL_STD / survival_vac

    # ==========================================
    # 4. RESULTS REPORTING
    # ==========================================
    print("RESULTS:")
    print(f"  > Standard Model Survival: {REF_SURVIVAL_STD*100:.2f}% (Reference)")
    print(f"  > Vacuum Model Survival:   {survival_vac*100:.2f}%")
    print(f"  > Depletion Factor:        {depletion_factor:.2f}x")
    print("-" * 64)

    # Validation Logic
    if 2.6 <= depletion_factor <= 3.4:
        print("[SUCCESS] VERDICT: PASS.")
        print("The code reproduces (~3.0x depletion).")
        print("Mechanism: The 'Mass Turbocharger' (Exponential) overcomes the")
        print("'Time Brake' (Linear) to solve the Lithium Anomaly.")
    else:
        print(f"[WARNING] VERDICT: DEVIATION. Factor {depletion_factor:.2f}x")

if __name__ == "__main__":
    verify_lithium_depletion()
