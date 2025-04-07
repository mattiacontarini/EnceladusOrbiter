
# Files and variables import
from OrbitPropagator import OrbitPropagator
from auxiliary import BenedikterInitialStates as Benedikter

# Tudat import
from tudatpy.data import save2txt
from tudatpy.interface import spice


# Define output file
output_path = "./nominal_orbits"

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

for i in range(len(initial_states)):
    initial_state = initial_states[i]

    # Propagate orbit
    [state_history, dependent_variable_history, computational_time] = UDP.retrieve_history(initial_state)

    # Save orbit to file
    save2txt(state_history,
             f"nominal_state_history_{i + 1}.dat",
             output_path)


