
# Files and variables import
from OrbitPropagator import OrbitPropagator
from auxiliary import BenedikterInitialStates as Benedikter

# Tudat import
from tudatpy.data import save2txt


# Define output file
output_path = "./kernels"

# Set up orbit propagator object with nominal settings
UDP = OrbitPropagator.from_config()

# Retrieve initial states
initial_states = [Benedikter.K1_initial_cartesian_state,
                  Benedikter.K2_initial_cartesian_state,
                  Benedikter.K3_initial_cartesian_state]

# Load SPICE kernels



for i in range(len(initial_states)):
    initial_state = initial_states[i]

    # Propagate orbit
    [state_history, dependent_variable_history, computational_time] = UDP.retrieve_history(initial_state)

    # Save orbit to file
    save2txt(state_history,
             f"nominal_state_history_{i}.dat",
             output_path)


