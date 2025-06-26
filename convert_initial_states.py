
# Files and variables import
from auxiliary import BenedikterInitialStates as Benedikter

# Tudat import
from tudatpy.astro import element_conversion

# Packages import
import numpy as np
import os

# Gravitational parameter of Enceladus - Park et al. (2024)
enceladus_gravitational_parameter = 7.210366688598896E+9

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

