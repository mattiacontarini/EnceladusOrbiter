
# Files and variables import
from auxiliary import BenedikterInitialStates as Benedikter
from auxiliary.utilities import utilities as Util

# Tudat import
from tudatpy.astro import element_conversion
from tudatpy import constants

# Packages import
import numpy as np
import os

# Gravitational parameter of Enceladus - Park et al. (2024)
enceladus_gravitational_parameter = 7.210366688598896E+9

# Radius of Enceladus - Porco et al. (2006)
enceladus_radius = 252.1e3

benedikter_states = [Benedikter.K1_initial_cartesian_state,
                     Benedikter.K2_initial_cartesian_state,
                     Benedikter.K3_initial_cartesian_state,]

output_path = "./output/initial_states_conversion"
os.makedirs(output_path, exist_ok=True)

# Convert initial states from Cartesian to Keplerian elements
for i in range(len(benedikter_states)):
    initial_cartesian_state = benedikter_states[i]
    initial_keplerian_state = element_conversion.cartesian_to_keplerian(initial_cartesian_state,
                                                                        enceladus_gravitational_parameter)
    file_path = os.path.join(output_path, f"initial_keplerian_state_K{i + 1}.txt")
    np.savetxt(file_path, initial_keplerian_state)


# Compute average altitude
simulation_duration = 28.0*constants.JULIAN_DAY
for i in range(len(benedikter_states)):
    state_history_cartesian = np.loadtxt(f"nominal_orbits/simulation_duration_{simulation_duration}/nominal_state_history_{i + 1}.dat")
    altitude_history = np.zeros((state_history_cartesian.shape[0], 1))
    for k in range(state_history_cartesian.shape[0]):
        altitude_history[k] = np.sqrt(state_history_cartesian[k, 1]**2 + state_history_cartesian[k, 2]**2 + state_history_cartesian[k, 3]**2) - enceladus_radius

    mean_altitude = np.mean(altitude_history)
    file_path = os.path.join(output_path, f"mean_altitude_K{i + 1}.txt")
    np.savetxt(file_path, [mean_altitude])


