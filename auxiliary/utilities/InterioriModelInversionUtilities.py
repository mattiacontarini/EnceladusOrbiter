# Files import
from auxiliary import InteriorModelInversionConfig as InteriorModelInvConfig

# General imports
import sys
sys.path.append("/Users/mattiacontarini/miniconda3/envs/tudat-bundle-fork/lib/python3.11/site-packages")
import numpy as np

def get_ocean_density(R_core, R_ocean, rho_shell):
    rho_mean = InteriorModelInvConfig.M_Enceladus / (4/3 * np.pi * InteriorModelInvConfig.R_Enceladus**3)

    num_1 = InteriorModelInvConfig.MoI_Enceladus
    num_2 = - 8/15*np.pi*R_core**2 * (rho_mean * InteriorModelInvConfig.R_Enceladus**3 - rho_shell * (InteriorModelInvConfig.R_Enceladus**3 - R_ocean**3))
    num_3 = - 8/15*np.pi*rho_shell*(InteriorModelInvConfig.R_Enceladus**5 - R_ocean**5)
    den = 8/15*np.pi*(R_ocean**5 - R_core**5 - R_core ** 2 * (R_ocean**3 - R_core**3))

    rho_ocean = (num_1 + num_2 + num_3) / den
    return rho_ocean

def get_core_density(R_core, R_ocean, rho_shell, rho_ocean):
    rho_mean = InteriorModelInvConfig.M_Enceladus / (4/3 * np.pi * InteriorModelInvConfig.R_Enceladus**3)

    num = (rho_mean * InteriorModelInvConfig.R_Enceladus**3 - rho_ocean * (R_ocean**3 - R_core**3) -
           rho_shell * (InteriorModelInvConfig.R_Enceladus**3 - R_ocean**3))
    den = R_core ** 3
    return num / den

def next_pow_two(n):
    i = 1
    while i < n:
        i = i << 1
    return i

def autocorr_func_1d(x, normalize=True):
    x = np.atleast_1d(x)
    if len(x.shape) != 1:
        raise ValueError("invalid dimensions for 1D autocorrelation function")
    n = next_pow_two(len(x))

    # Compute the FFT and then (from that) the auto-correlation function
    f = np.fft.fft(x - np.mean(x), n=2 * n)
    acf = np.fft.ifft(f * np.conjugate(f))[: len(x)].real
    acf /= 4 * n

    # Optionally normalize
    if normalize:
        acf /= acf[0]

    return acf

def autocorr_new(y, c=5.0):
    f = np.zeros(y.shape[1])
    for yy in y:
        f += autocorr_func_1d(yy)
    f /= len(y)
    taus = 2.0 * np.cumsum(f) - 1.0
    window = auto_window(taus, c)
    return taus[window]

# Automated windowing procedure following Sokal (1989)
def auto_window(taus, c):
    m = np.arange(len(taus)) < c * taus
    if np.any(m):
        return np.argmin(m)
    return len(taus) - 1

def check_convergence(tau, N):
    if N > 50 * tau:
        convergence_check = True
    else:
        convergence_check = False
    return convergence_check