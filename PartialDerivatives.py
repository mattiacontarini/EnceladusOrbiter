
# Tudat import
from tudatpy import constants

# Packages import
import sympy as sym
import numpy as np

#######################################################################################################################
### Define symbolic variables
#######################################################################################################################

alpha = sym.Symbol('alpha')
delta = sym.Symbol('delta')
W0 = sym.Symbol('W0')
W0_dot = sym.Symbol('W0_dot')
phi = sym.Symbol('phi')
gamma = sym.Symbol('gamma')
t = sym.Symbol('t')

omega = 1.37 * constants.JULIAN_DAY

nb_nutation_terms = dict(
    alpha=15,
    delta=14,
    W=18
)

# Set up expression for longitude of prime meridian
W = W0 + W0_dot*t + nutation_terms + phi*sym.sin(omega*t + gamma)

R = [
    [-sym.cos(W)*sym.sin(alpha)-sym.cos(alpha)*sym.sin(delta)*sym.sin(W), sym.sin(alpha)*sym.sin(delta)*sym.sin(W)-sym.cos(alpha)*sym.cos(W), sym.cos(delta)*sym.sin(W)],
    [np.cos(alpha)*np.cos(W)*np.sin(delta) - np.sin(alpha)*np.sin(W), -np.cos(W)*np.sin(alpha)*np.sin(delta)-np.cos(alpha)*np.sin(W), -np.cos(delta)*np.cos(W)],
    [np.cos(alpha)*np.cos(delta), -np.cos(delta)*np.sin(alpha), np.sin(delta)]
]

dR_dphi = R.diff(phi)
