import numpy as np
import pandas as pd
import requests
import io
from scipy.integrate import quad

# 1. Define Planck 2018 Cosmology Parameters
H0_PLANCK = 67.4
OM_PLANCK = 0.315
OL_PLANCK = 1.0 - OM_PLANCK
C_LIGHT = 299792.458  # km/s

def hubble_inverse(z):
    """Inverse of the dimensionless Hubble parameter E(z)"""
    return 1.0 / np.sqrt(OM_PLANCK * (1 + z)**3 + OL_PLANCK)

def get_planck_mu(z):
    """Calculate Distance Modulus for Planck Cosmology"""
    if z <= 0: return np.nan
    # Luminosity distance: DL = (1+z) * (c/H0) * integral(1/E(z))
    integral, _ = quad(hubble_inverse, 0, z)
    d_L = (1 + z) * (C_LIGHT / H0_PLANCK) * integral
    return 5 * np.log10(d_L) + 25

# 2. Load Pantheon+ Data
url = "https://raw.githubusercontent.com/PantheonPlusSH0ES/DataRelease/main/Pantheon%2B_Data/4_DISTANCES_AND_COVAR/Pantheon%2BSH0ES.dat"
print("Downloading data...")
response = requests.get(url)
data = pd.read_csv(io.StringIO(response.text), sep=r'\s+')

# 3. Process Data
# Filter out non-positive redshifts
df = data[data['zHD'] > 0.01].copy()

# Calculate Theoretical Planck Brightness
df['mu_Planck'] = df['zHD'].apply(get_planck_mu)

# Calculate Residual (Observed - Planck)
df['residual'] = df['MU_SH0ES'] - df['mu_Planck']

# 4. Filter for High-Redshift Bin (z > 0.65)
high_z_bin = df[df['zHD'] > 0.65]

# 5. Calculate and Print Statistics
mean_resid = high_z_bin['residual'].mean()
count = len(high_z_bin)

print("-" * 30)
print(f"RESULTS FOR z > 0.65")
print("-" * 30)
print(f"Count: {count}")
print(f"Mean Residual: {mean_resid:.4f} mag")
print("-" * 30)
