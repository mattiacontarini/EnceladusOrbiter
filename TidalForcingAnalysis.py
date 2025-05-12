
#######################################################################################################################
### Import statements #################################################################################################
#######################################################################################################################

# Tudat import
from tudatpy import constants
from tudatpy.interface import spice
from tudatpy.data import save2txt
from tudatpy.astro.element_conversion import cartesian_to_keplerian

# General import
import numpy as np
import math
import sympy as sym
import os

def kronecker_delta(order):
     if order == 0:
         delta = 1.0
     else:
        delta = 0.0

     return delta


def get_normalised_legendre_function(degree, order, arg):
    normalisation_factor = np.sqrt(
        math.factorial(degree - order) * (2 * degree + 1) * (2 - kronecker_delta(order)) / math.factorial(degree + order)
    )

    mu_sym = sym.symbols('mu')
    f_sym = (mu_sym ** 2 - 1) ** degree
    derivative = f_sym.diff(mu_sym, order + degree)
    derivative = derivative.subs(mu_sym, arg)

    unnormalised_legendre_function = (
            ( (1 - arg**2) ** (order / 2) ) * (1 / (2 ** degree * math.factorial(degree))) * derivative
                                     )
    normalised_legendre_function = normalisation_factor * unnormalised_legendre_function

    return normalised_legendre_function


def main():
    #######################################################################################################################
    ### Input parameters ##################################################################################################
    #######################################################################################################################

    arc_start = 0
    arc_duration = 60 * constants.JULIAN_DAY
    epochs_step = 15

    # Gravitational parameter of Enceladus
    enceladus_gravitational_parameter = 7.210366688598896E+9  # From Park et al., 2024

    # Gravitational parameter of Saturn
    saturn_gravitational_parameter = 3.793120749865224E+16  # From Iess et al., 2019

    # Value of the k2 Love number for Enceladus
    enceladus_k2_love_number = complex(0.02, 0.01)  # From Genova et al., 2024

    # Degree of the Love number to study
    degrees_to_consider = [2]

    # Order to consider
    order = 0

    # Colatitude of the point to consider for computing the tidal forcing
    colatitude = np.deg2rad(90)

    # Longitude of the point to consider for computing the tidal forcing
    longitude = np.deg2rad(0)

    # Set output directory
    output_directory = "./output/tidal_forcing_analysis"

    ###################################################################################################################
    ### Compute tidal forcing history #################################################################################
    ###################################################################################################################

    # Generate vector of epochs to study
    arc_end = arc_start + arc_duration
    epochs = np.arange(arc_start, arc_end, epochs_step)
    nb_epochs = len(epochs)

    # Load SPICE kernels
    spice.load_standard_kernels()
    kernels_to_load = [
        "/Users/mattiacontarini/Documents/Code/Thesis/kernels/de440.bsp",
        "/Users/mattiacontarini/Documents/Code/Thesis/kernels/sat441l.bsp",
    ]
    spice.load_standard_kernels(kernels_to_load)

    # Retrieve average radius of Enceladus
    enceladus_average_radius = spice.get_average_radius("Enceladus")

    # Compute argument of Legendre polynomials
    arg = np.cos(colatitude)

    # Compute history of tidal forcing for every degree
    for degree in degrees_to_consider:

        # Compute normalised Legendre function
        normalised_legendre_function = get_normalised_legendre_function(degree, order, arg)

        # Create output directory
        output_path = os.path.join(output_directory, f"degree_{degree}")
        os.makedirs(output_path, exist_ok=True)

        distance_history = dict()
        kepler_elements_history = dict()
        tidal_forcing = dict()
        for epoch in epochs:

            # Retrieve Cartesian state of Enceladus wrt Saturn
            enceladus_cartesian_state = spice.get_body_cartesian_state_at_epoch(
                "Enceladus",
                "Saturn",
                "J2000",
                "NONE",
                epoch
            )

            # Retrieve Cartesian position of Enceladus wrt Saturn
            enceladus_cartesian_position = enceladus_cartesian_state[:3]

            # Compute distance between Saturn and Enceladus
            distance = np.linalg.norm(enceladus_cartesian_position)
            distance_history[epoch] = distance

            # Convert Cartesian state to Keplerian elements
            kepler_elements = cartesian_to_keplerian(enceladus_cartesian_state,
                                                     saturn_gravitational_parameter)
            kepler_elements_history[epoch] = kepler_elements

            # Compute tidal forcing at current epoch
            tidal_forcing[epoch] = []

            tidal_forcing[epoch] = []
            tidal_forcing_cosine = (( enceladus_k2_love_number.real / (2 * degree + 1) ) *
                                    ( saturn_gravitational_parameter / enceladus_gravitational_parameter ) *
                                    ( enceladus_average_radius / distance ) ** (degree + 1) *
                                    normalised_legendre_function  * np.cos(order * longitude))
            tidal_forcing[epoch].append( tidal_forcing_cosine )

            tidal_forcing_sine = (( enceladus_k2_love_number.real / (2 * degree + 1) ) *
                                    ( saturn_gravitational_parameter / enceladus_gravitational_parameter ) *
                                    ( enceladus_average_radius / distance ) ** (degree + 1) *
                                    normalised_legendre_function * np.sin(order * longitude))
            tidal_forcing[epoch].append( tidal_forcing_sine )

        # Save tidal forcing and variables history to file
        save2txt(tidal_forcing,
                 "tidal_forcing_history.dat",
                 output_path)
        save2txt(distance_history,
                 "distance_history.dat",
                 output_path)
        save2txt(kepler_elements_history,
                 "kepler_elements_history.dat",
                 output_path)



if __name__ == "__main__":
    main()
