"""
Select accelerations acting on vehicle.
"""

# Files and variables import
from OrbitPropagator import OrbitPropagator
from auxiliary import BenedikterInitialStates as Benedikter
from auxiliary import utilities as Util
from auxiliary import OrbitPropagatorConfig as PropConfig

# Tudat import
from tudatpy import numerical_simulation
from tudatpy.data import save2txt
from tudatpy.util import result2array
from tudatpy import constants
from tudatpy.interface import spice

# Packages import
import numpy as np
import datetime
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import os


#######################################################################################################################
###
#######################################################################################################################
def main():
    # Retrieve current time stamp
    time_stamp = datetime.datetime.now().strftime("%Y.%m.%d.%H.%M.%S")

    # Define output folder
    output_folder = "./output/accelerations_selection"

    # Build output_path
    output_path = os.path.join(output_folder, time_stamp)
    os.makedirs(output_path, exist_ok=True)

    study_physical_model_error = True
    if study_physical_model_error:

        # Load SPICE kernels for simulation
        spice.load_standard_kernels()
        kernels_to_load = ["./kernels/de438.bsp", "./kernels/sat427.bsp"]
        spice.load_standard_kernels(kernels_to_load)

        bodies_to_study = ["Sun",
                           "Mimas",
                           "Tethys",
                           "Dione",
                           "Rhea",
                           "Titan",
                           "Jupiter_barycenter",
                           "Uranus_barycenter",
                           "Neptune_barycenter",
                           "Mars_barycenter",
                           "Earth_barycenter",
                           "Venus",
                           "Mercury"]

        barycenters_list = ["Jupiter_barycenter",
                            "Mars_barycenter",
                            "Earth_barycenter",
                            "Uranus_barycenter",
                            "Neptune_barycenter"]

        bodies_to_create = ["Sun",
                            "Enceladus",
                            "Saturn",
                            "Mimas",
                            "Tethys",
                            "Dione",
                            "Rhea",
                            "Titan",
                            "Venus",
                            "Mercury"]

        acceleration_settings_list = dict(
            Sun=[numerical_simulation.propagation_setup.acceleration.point_mass_gravity(),
                 numerical_simulation.propagation_setup.acceleration.radiation_pressure(), ],
            Mimas=[numerical_simulation.propagation_setup.acceleration.point_mass_gravity(), ],
            Tethys=[numerical_simulation.propagation_setup.acceleration.point_mass_gravity(), ],
            Dione=[numerical_simulation.propagation_setup.acceleration.point_mass_gravity(), ],
            Rhea=[numerical_simulation.propagation_setup.acceleration.point_mass_gravity(), ],
            Titan=[numerical_simulation.propagation_setup.acceleration.point_mass_gravity(), ],
            Jupiter_barycenter=[numerical_simulation.propagation_setup.acceleration.point_mass_gravity(), ],
            Uranus_barycenter=[numerical_simulation.propagation_setup.acceleration.point_mass_gravity(), ],
            Neptune_barycenter=[numerical_simulation.propagation_setup.acceleration.point_mass_gravity(), ],
            Mars_barycenter=[numerical_simulation.propagation_setup.acceleration.point_mass_gravity(), ],
            Earth_barycenter=[numerical_simulation.propagation_setup.acceleration.point_mass_gravity(), ],
            Venus=[numerical_simulation.propagation_setup.acceleration.point_mass_gravity(), ],
            Mercury=[numerical_simulation.propagation_setup.acceleration.point_mass_gravity(), ],
        )

        case_keys = dict(
            Sun=["GM", "SRP"],
            Mimas=["GM"],
            Tethys=["GM"],
            Dione=["GM"],
            Rhea=["GM"],
            Titan=["GM"],
            Jupiter_barycenter=["GM"],
            Uranus_barycenter=["GM"],
            Neptune_barycenter=["GM"],
            Mars_barycenter=["GM"],
            Earth_barycenter=["GM"],
            Venus=["GM"],
            Mercury=["GM"],
        )

        UDP = OrbitPropagator.from_config()

        dependent_variables_to_save = [
            numerical_simulation.propagation_setup.dependent_variable.altitude("Vehicle", "Enceladus"),
            numerical_simulation.propagation_setup.dependent_variable.single_acceleration_norm(
                numerical_simulation.propagation_setup.acceleration.spherical_harmonic_gravity_type,
                "Vehicle",
                "Enceladus"
            ),
            numerical_simulation.propagation_setup.dependent_variable.single_acceleration_norm(
                numerical_simulation.propagation_setup.acceleration.spherical_harmonic_gravity_type,
                "Vehicle",
                "Saturn"
            ),
        ]

        UDP.dependent_variables_to_save = dependent_variables_to_save

        initial_state = Benedikter.K1_initial_cartesian_state

        [benchmark_state_history, benchmark_dependent_variable_history,
         benchmark_computational_time] = UDP.retrieve_history(initial_state)

        save2txt(benchmark_state_history,
                 "state_history_benchmark.dat",
                 output_path
                 )
        save2txt(benchmark_dependent_variable_history,
                 "dependent_variable_history_benchmark.dat",
                 output_path
                 )

        acceleration_settings_on_vehicle = PropConfig.acceleration_settings_on_vehicle
        for body in bodies_to_study:

            nb_accelerations_to_study = len(acceleration_settings_list[body])
            for i in range(nb_accelerations_to_study):
                acceleration = acceleration_settings_list[body][i]

                # Re-initialize propagator object
                UDP = OrbitPropagator.from_config()
                UDP.barycenters_list = barycenters_list
                UDP.bodies_to_create = bodies_to_create

                acceleration_settings_on_vehicle[body] = [acceleration]
                UDP.acceleration_settings_on_vehicle = acceleration_settings_on_vehicle

                if body == "Sun" and i == 1:
                    dependent_variables_to_save = [
                        numerical_simulation.propagation_setup.dependent_variable.altitude("Vehicle", "Enceladus"),
                        numerical_simulation.propagation_setup.dependent_variable.single_acceleration_norm(
                            numerical_simulation.propagation_setup.acceleration.cannonball_radiation_pressure_type,
                            "Vehicle",
                            body
                        )
                    ]
                else:
                    dependent_variables_to_save = [
                        numerical_simulation.propagation_setup.dependent_variable.altitude("Vehicle",
                                                                                           "Enceladus"),
                        numerical_simulation.propagation_setup.dependent_variable.single_acceleration_norm(
                            numerical_simulation.propagation_setup.acceleration.point_mass_gravity_type,
                            "Vehicle",
                            body
                        )
                    ]

                UDP.dependent_variables_to_save = dependent_variables_to_save

                [state_history, dependent_variable_history, computational_time] = UDP.retrieve_history(initial_state)

                case = case_keys[body][i]
                save2txt(state_history,
                         f"state_history_{body}_case_{case}.dat",
                         output_path
                         )
                save2txt(dependent_variable_history,
                         f"dependent_variable_history_{body}_case_{case}.dat",
                         output_path
                         )

                state_history_difference = Util.compute_benchmarks_state_history_difference(state_history,
                                                                                            benchmark_state_history,
                                                                                            f"benchmark_state_history_difference_{body}_case_{case}.dat",
                                                                                            output_path)

                # Prepare for next body and delete current acceleration
                del acceleration_settings_on_vehicle[body]

        # Initialize propagator object
        UDP = OrbitPropagator.from_config()

        initial_state = Benedikter.K1_initial_cartesian_state

        # Propagate case with only point mass of saturn
        acceleration_settings_on_vehicle = PropConfig.acceleration_settings_on_vehicle
        acceleration_settings_on_vehicle["Saturn"] = [
            numerical_simulation.propagation_setup.acceleration.point_mass_gravity()]
        dependent_variables_to_save = [
            numerical_simulation.propagation_setup.dependent_variable.altitude("Vehicle", "Enceladus"),
            numerical_simulation.propagation_setup.dependent_variable.single_acceleration(
                numerical_simulation.propagation_setup.acceleration.point_mass_gravity_type,
                "Vehicle",
                "Saturn"
            )
        ]
        del UDP.acceleration_settings_on_vehicle["Saturn"]

        UDP.acceleration_settings_on_vehicle["Saturn"] = [numerical_simulation.propagation_setup.acceleration.point_mass_gravity()]
        UDP.dependent_variables_to_save = dependent_variables_to_save
        [state_history_GM, dependent_variable_history_GM, computational_time_GM] = UDP.retrieve_history(initial_state)
        save2txt(state_history_GM,
                 "state_history_study_Saturn_case_GM.dat",
                 output_path)
        save2txt(dependent_variable_history_GM,
                 "dependent_variable_history_study_Saturn_GM.dat",
                 output_path)

        # Propagate case with spherical harmonics of Saturn
        UDP_SH = OrbitPropagator.from_config()
        UDP_SH.acceleration_settings_on_vehicle["Saturn"] = [
            numerical_simulation.propagation_setup.acceleration.spherical_harmonic_gravity(8, 8)]
        dependent_variables_to_save = [
            numerical_simulation.propagation_setup.dependent_variable.altitude("Vehicle", "Enceladus"),
            numerical_simulation.propagation_setup.dependent_variable.single_acceleration(
                numerical_simulation.propagation_setup.acceleration.spherical_harmonic_gravity_type,
                "Vehicle",
                "Saturn"
            )
        ]
        UDP_SH.dependent_variables_to_save = dependent_variables_to_save
        [state_history_SH, dependent_variable_history_SH, computational_time] = UDP_SH.retrieve_history(initial_state)
        save2txt(state_history_SH,
                 "state_history_study_Saturn_case_SH.dat",
                 output_path)
        save2txt(dependent_variable_history_SH,
                 "dependent_variable_history_study_Saturn_case_SH.dat",
                 output_path)

        # Compute state and acceleration difference
        state_history_difference = Util.compute_benchmarks_state_history_difference(state_history_GM,
                                                                                    state_history_SH,
                                                                                    "state_history_difference_study_Saturn.dat",
                                                                                    output_path)
        epochs = list(state_history_GM.keys())
        acceleration_difference_history = dict()
        for epoch in epochs:
            acceleration_difference_history[epoch] = (dependent_variable_history_GM[epoch][1:4] -
                                                      dependent_variable_history_SH[epoch][1:4])

        save2txt(acceleration_difference_history,
                 "acceleration_history_difference_study_Saturn.dat",
                 output_path)


if __name__ == "__main__":
    main()
