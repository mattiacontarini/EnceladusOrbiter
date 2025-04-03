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
    study_physical_model_error = True
    if study_physical_model_error:

        # Retrieve current time stamp
        time_stamp = datetime.datetime.now().strftime("%Y.%m.%d.%H.%M.%S")

        # Define output folder
        output_folder = "./output/accelerations_selection"

        # Build output_path
        output_path = os.path.join(output_folder, time_stamp)
        os.makedirs(output_path, exist_ok=True)

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
                 numerical_simulation.propagation_setup.acceleration.cannonball_radiation_pressure(), ],
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

        UDP = OrbitPropagator.from_config()

        dependent_variables_to_save = [
            numerical_simulation.propagation_setup.dependent_variable.altitude("Vehicle", "Enceladus"),
            #numerical_simulation.propagation_setup.dependent_variable.single_acceleration_norm(
            #    numerical_simulation.propagation_setup.acceleration.point_mass_gravity_type,
            #    "Vehicle",
            #    "Enceladus"
            #),
            numerical_simulation.propagation_setup.dependent_variable.single_acceleration_norm(
                numerical_simulation.propagation_setup.acceleration.spherical_harmonic_gravity_type,
                "Vehicle",
                "Enceladus"
            ),
            #numerical_simulation.propagation_setup.dependent_variable.single_acceleration_norm(
            #    numerical_simulation.propagation_setup.acceleration.point_mass_gravity_type,
            #    "Vehicle",
            #    "Saturn"
            #),
            numerical_simulation.propagation_setup.dependent_variable.single_acceleration_norm(
                numerical_simulation.propagation_setup.acceleration.spherical_harmonic_gravity_type,
                "Vehicle",
                "Saturn"
            ),
        ]

        bodies_color_code = dict(
            Sun="orange",
            Enceladus="blue",
            Saturn="red",
            Mimas="green",
            Tethys="olive",
            Dione="cyan",
            Rhea="gray",
            Titan="purple",
            Jupiter_barycenter="black",
            Uranus_barycenter="pink",
            Neptune_barycenter="m",
            Mars_barycenter="darkred",
            Earth_barycenter="slateblue",
            Venus="yellow",
            Mercury="gold",
        )

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

            # Re-initialize propagator object
            UDP = OrbitPropagator.from_config()
            UDP.barycenters_list = barycenters_list
            UDP.bodies_to_create = bodies_to_create

            acceleration_settings_on_vehicle[body] = acceleration_settings_list[body]
            UDP.acceleration_settings_on_vehicle = acceleration_settings_on_vehicle

            dependent_variables_to_save = [
                numerical_simulation.propagation_setup.dependent_variable.altitude("Vehicle",
                                                                                   "Enceladus"),
                numerical_simulation.propagation_setup.dependent_variable.single_acceleration_norm(
                    numerical_simulation.propagation_setup.acceleration.point_mass_gravity_type,
                    "Vehicle",
                    body
                )
            ]

            if body == "Sun":
                dependent_variables_to_save = [
                    numerical_simulation.propagation_setup.dependent_variable.altitude("Vehicle", "Enceladus"),
                    numerical_simulation.propagation_setup.dependent_variable.single_acceleration_norm(
                        numerical_simulation.propagation_setup.acceleration.point_mass_gravity_type,
                        "Vehicle",
                        body
                    ),
                    numerical_simulation.propagation_setup.dependent_variable.single_acceleration_norm(
                        numerical_simulation.propagation_setup.acceleration.cannonball_radiation_pressure_type,
                        "Vehicle",
                        body
                    )
                ]

            UDP.dependent_variables_to_save = dependent_variables_to_save

            [state_history, dependent_variable_history, computational_time] = UDP.retrieve_history(initial_state)

            save2txt(state_history,
                     f"state_history_{body}_case.dat",
                     output_path
                     )
            save2txt(dependent_variable_history,
                     f"dependent_variable_history_{body}_case.dat",
                     output_path
                     )

            state_history_difference = Util.compute_benchmarks_state_history_difference(state_history,
                                                                                        benchmark_state_history,
                                                                                        f"benchmark_state_history_difference_{body}_case.dat",
                                                                                        output_path)

            # Prepare for next body and delete current acceleration
            del acceleration_settings_on_vehicle[body]


    study_delta_v_corrections_flag = False
    if study_delta_v_corrections_flag:
        ...


if __name__ == "__main__":
    main()
