"""
Order of interior parameters:
 0. rho_shell
 1. mu_shell
 2. rho_ocean
 3. R_ocean
 4. mu_core

 Order of observations:
 0. k2_real
 1. k2_imag
 2. h2
 3. libration
"""

# Files import
from InteriorModelInversionObject import InteriorModelInversion


UDP = InteriorModelInversion.from_config()
observations = UDP.compute_observations(x=1)


