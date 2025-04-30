"""
File used to define the OrbitPropagator class.
The OrbitPropagator class is used to propagate the selected initial condition for the spacecraft.
"""

# Files and variables import
from auxiliary import VehicleParameters as VehicleParam
from auxiliary import OrbitPropagatorConfig as PropConfig
from auxiliary.utilities import environment_setup_utilities as EnvUtil
from auxiliary import CovarianceAnalysisConfig as CovAnalysisConfig

# Tudat import
from tudatpy import numerical_simulation
from tudatpy.numerical_simulation import environment_setup
from tudatpy.numerical_simulation import propagation_setup
from tudatpy.astro import element_conversion
from tudatpy.kernel.interface import spice_interface, spice
from tudatpy import constants


class OrbitPropagator:
    """
    Class to initialise and simulate a stable orbit around Enceladus.
    The class is created specifically for this problem.
    """

    def __init__(self,
                 simulation_start_epoch,
                 simulation_end_epoch,
                 termination_altitude,
                 bodies_to_create,
                 barycenters_list,
                 acceleration_settings_on_vehicle,
                 integrator_settings,
                 dependent_variables_to_save,
                 ):
        # Simulation parameters
        self.simulation_start_epoch = simulation_start_epoch
        self.simulation_end_epoch = simulation_end_epoch
        self.termination_altitude = termination_altitude
        self.bodies_to_create = bodies_to_create
        self.barycenters_list = barycenters_list
        self.acceleration_settings_on_vehicle = acceleration_settings_on_vehicle
        self.integrator_settings = integrator_settings
        self.dependent_variables_to_save = dependent_variables_to_save

    @classmethod
    def from_config(cls):
        simulation_start_epoch = PropConfig.simulation_start_epoch
        simulation_end_epoch = PropConfig.simulation_end_epoch
        termination_altitude = PropConfig.termination_altitude
        bodies_to_create = PropConfig.bodies_to_create
        barycenters_list = PropConfig.barycenters_list
        acceleration_settings_on_vehicle = PropConfig.acceleration_settings_on_vehicle
        integrator_settings = PropConfig.integrator_settings
        dependent_variables_to_save = PropConfig.dependent_variables_to_save

        return cls(simulation_start_epoch,
                   simulation_end_epoch,
                   termination_altitude,
                   bodies_to_create,
                   barycenters_list,
                   acceleration_settings_on_vehicle,
                   integrator_settings,
                   dependent_variables_to_save
                   )

    def retrieve_history(self,
                         initial_state,
                         ):
        # Create settings for celestial bodies
        global_frame_origin = "Enceladus"
        global_frame_orientation = "J2000"

        # Retrieve default body settings
        body_settings = environment_setup.get_default_body_settings(self.bodies_to_create,
                                                                    global_frame_origin,
                                                                    global_frame_orientation)

        # Add Jupiter barycenter
        for barycenter in self.barycenters_list:
            if barycenter == "Earth_barycenter":
                NAIF_ID = "3"
            elif barycenter == "Mars_barycenter":
                NAIF_ID = "4"
            elif barycenter == "Jupiter_barycenter":
                NAIF_ID = "5"
            elif barycenter == "Saturn_barycenter":
                NAIF_ID = "6"
            elif barycenter == "Uranus_barycenter":
                NAIF_ID = "7"
            elif barycenter == "Neptune_barycenter":
                NAIF_ID = "8"

            body_settings.add_empty_settings(barycenter)
            body_settings.get(barycenter).gravity_field_settings = environment_setup.gravity_field.central(
                spice_interface.get_body_gravitational_parameter(NAIF_ID)
            )
            body_settings.get(barycenter).ephemeris_settings = environment_setup.ephemeris.direct_spice(
                frame_origin=global_frame_origin,
                frame_orientation=global_frame_orientation,
                body_name_to_use=NAIF_ID
            )

        # Set rotation model settings for Enceladus
        synodic_rotation_rate_enceladus = EnvUtil.get_synodic_rotation_model_enceladus(self.simulation_start_epoch)
        initial_orientation_enceladus = spice.compute_rotation_matrix_between_frames("J2000",
                                                                                     "IAU_Enceladus",
                                                                                     self.simulation_start_epoch)
        body_settings.get("Enceladus").rotation_model_settings = environment_setup.rotation_model.simple(
            "J2000", "IAU_Enceladus", initial_orientation_enceladus,
            self.simulation_start_epoch, synodic_rotation_rate_enceladus)

        # Set gravity field settings for Enceladus
        body_settings.get("Enceladus").gravity_field_settings = EnvUtil.get_gravity_field_settings_enceladus_park(
            CovAnalysisConfig.maximum_degree_gravity_enceladus
        )
        body_settings.get(
            "Enceladus").gravity_field_settings.scaled_mean_moment_of_inertia = 0.335  # From Iess et al. (2014)

        # Set gravity field settings for Saturn
        body_settings.get("Saturn").gravity_field_settings = EnvUtil.get_gravity_field_settings_saturn_iess()
        body_settings.get(
            "Saturn").gravity_field_settings.scaled_mean_moment_of_inertia = 0.210  # From NASA Saturn Fact Sheet

        # Define bodies that are propagated and their central bodies of propagation
        bodies_to_propagate = ["Vehicle"]
        central_bodies = ["Enceladus"]

        # Create global accelerations dictionary
        acceleration_settings = {"Vehicle": self.acceleration_settings_on_vehicle}

        [state_history, dependent_variables_history, computational_time] = self.propagate_orbit(initial_state,
                                                                                                body_settings,
                                                                                                bodies_to_propagate,
                                                                                                central_bodies,
                                                                                                acceleration_settings,
                                                                                                self.integrator_settings,
                                                                                                self.dependent_variables_to_save)

        return [state_history, dependent_variables_history, computational_time]

    def propagate_orbit(
            self,
            initial_state,
            body_settings,
            bodies_to_propagate,
            central_bodies,
            acceleration_settings,
            integrator_settings,
            dependent_variables_to_save,
    ):
        # Create environment
        bodies = environment_setup.create_system_of_bodies(body_settings)

        # Create vehicle object and add properties
        bodies.create_empty_body("Vehicle")
        bodies.get("Vehicle").mass = VehicleParam.mass

        # Create aerodynamic coefficient settings
        reference_area = VehicleParam.drag_reference_area
        drag_coefficient = VehicleParam.drag_coefficient
        aero_coefficient_settings = environment_setup.aerodynamic_coefficients.constant(
            reference_area, [drag_coefficient, 0.0, 0.0]
        )

        # Add the aerodynamic interface to the environment
        environment_setup.add_aerodynamic_coefficient_interface(bodies, "Vehicle", aero_coefficient_settings)

        # Create radiation pressure settings
        reference_area = VehicleParam.radiation_pressure_reference_area
        radiation_pressure_coefficient = VehicleParam.radiation_pressure_coefficient
        occulting_bodies = ["Enceladus"]
        radiation_pressure_settings = environment_setup.radiation_pressure.cannonball(
            "Sun", reference_area, radiation_pressure_coefficient, occulting_bodies
        )

        # Add the radiation pressure interface to the environment
        environment_setup.add_radiation_pressure_interface(bodies, "Vehicle", radiation_pressure_settings)

        # Create acceleration models
        acceleration_models = propagation_setup.create_acceleration_models(
            bodies, acceleration_settings, bodies_to_propagate, central_bodies
        )

        # Create termination settings
        time_termination_settings = propagation_setup.propagator.time_termination(self.simulation_end_epoch)

        altitude_termination_variable = propagation_setup.dependent_variable.altitude(
            "Vehicle", "Enceladus"
        )
        altitude_termination_settings = propagation_setup.propagator.dependent_variable_termination(
            dependent_variable_settings=altitude_termination_variable,
            limit_value=self.termination_altitude,
            use_as_lower_limit=True,
            terminate_exactly_on_final_condition=False
        )
        termination_settings_list = [time_termination_settings, altitude_termination_settings]
        termination_settings = propagation_setup.propagator.hybrid_termination(
            termination_settings_list, fulfill_single_condition=True
        )

        # Create propagation settings
        propagator_settings = propagation_setup.propagator.translational(
            central_bodies,
            acceleration_models,
            bodies_to_propagate,
            initial_state,
            self.simulation_start_epoch,
            integrator_settings,
            termination_settings,
            output_variables=dependent_variables_to_save,
        )

        # Create simulation object and propagate dynamics
        dynamics_simulator = numerical_simulation.create_dynamics_simulator(
            bodies, propagator_settings
        )

        # Retrieve simulation results
        propagation_results = dynamics_simulator.propagation_results

        # Extract numerical solution for states and dependent variables
        state_history = propagation_results.state_history
        dependent_variables_history = propagation_results.dependent_variable_history
        computational_time = propagation_results.total_computation_time

        return [state_history, dependent_variables_history, computational_time]
