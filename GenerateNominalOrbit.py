
# Files and variables import
from OrbitPropagator import OrbitPropagator
from auxiliary import BenedikterInitialStates as Benedikter

# Tudat import
from tudatpy.data import save2txt
from tudatpy.interface import spice
from tudatpy import constants

# Packages import
import os


# Define output file
output_folder = "./nominal_orbits"

simulation_durations = [28.0 * constants.JULIAN_DAY,
                        60.0 * constants.JULIAN_DAY,
                        180.0 * constants.JULIAN_DAY,
                        1.0 * constants.JULIAN_YEAR]

# Set up orbit propagator object with nominal settings
UDP = OrbitPropagator.from_config()

# Retrieve initial states
initial_states = [Benedikter.K1_initial_cartesian_state,
                  Benedikter.K2_initial_cartesian_state,
                  Benedikter.K3_initial_cartesian_state]

# Load SPICE kernels
spice.load_standard_kernels()
kernels_to_load = ["/Users/mattiacontarini/Documents/Code/Thesis/kernels/de438.bsp",
                   "/Users/mattiacontarini/Documents/Code/Thesis/kernels/sat427.bsp"]
spice.load_standard_kernels(kernels_to_load)

for simulation_duration in simulation_durations:

    output_path = os.path.join(output_folder, f"simulation_duration_{simulation_duration}")
    os.makedirs(output_path, exist_ok=True)

    UDP.simulation_end_epoch = UDP.simulation_start_epoch + simulation_duration

    for i in range(len(initial_states)):
        initial_state = initial_states[i]

        # Propagate orbit
        [state_history, dependent_variable_history, computational_time] = UDP.retrieve_history(initial_state)

        # Save orbit to file
        save2txt(state_history,
                f"nominal_state_history_{i + 1}.dat",
                output_path)

        save2txt(dependent_variable_history,
                f"nominal_dependent_variable_history_{i + 1}.dat",
                 output_path)


