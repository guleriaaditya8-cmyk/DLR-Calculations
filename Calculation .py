import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt

kb = 8.617333262145e-5   # Boltzmann constant in eV/K
t = 293                  # Temperature in Kelvin
E0 = kb * t              # Average energy in eV

def dn_n0(E):
    return (2/np.sqrt(np.pi)) * (np.sqrt(E) / (E0**1.5)) * np.exp(-E / E0)

result, error = quad(dn_n0, 0, np.inf)

print("Integral of DN/N0 over all energies:", result)
print("Error:", error)

E = np.linspace(0, 1, 500)   # Energy range in eV
spectrum = dn_n0(E)

plt.plot(E, spectrum)
plt.xlabel("Energy (eV)")
plt.ylabel("Probability Density DN/N0")
plt.title("Maxwell-Boltzmann Energy Spectrum of Neutrons at 293 K")
plt.grid()

plt.show()