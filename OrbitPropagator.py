"""
File used to define the OrbitPropagator class.
The OrbitPropagator class is used to propagate the selected initial condition for the spacecraft.
"""

# Files and variables import
from auxiliary import VehicleParameters as VehicleParam
from auxiliary import OrbitPropagatorConfig as PropConfig
from auxiliary import utilities as Util

# Tudat import
from tudatpy import numerical_simulation
from tudatpy.numerical_simulation import environment_setup
from tudatpy.numerical_simulation import propagation_setup
from tudatpy.kernel.interface import spice_interface


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
                 Jupiter_barycenter_flag,
                 acceleration_settings_on_vehicle,
                 integrator_settings
                 ):
        # Simulation parameters
        self.simulation_start_epoch = simulation_start_epoch
        self.simulation_end_epoch = simulation_end_epoch
        self.termination_altitude = termination_altitude
        self.bodies_to_create = bodies_to_create
        self.Jupiter_barycenter_flag = Jupiter_barycenter_flag
        self.acceleration_settings_on_vehicle = acceleration_settings_on_vehicle
        self.integrator_settings = integrator_settings

    @classmethod
    def from_config(cls):
        simulation_start_epoch = PropConfig.simulation_start_epoch
        simulation_end_epoch = PropConfig.simulation_end_epoch
        termination_altitude = PropConfig.termination_altitude
        bodies_to_create = PropConfig.bodies_to_create
        Jupiter_barycenter_flag = PropConfig.Jupiter_barycenter_flag
        acceleration_settings_on_vehicle = PropConfig.acceleration_settings_on_vehicle
        integrator_settings = PropConfig.integrator_settings

        return cls(simulation_start_epoch,
                   simulation_end_epoch,
                   termination_altitude,
                   bodies_to_create,
                   Jupiter_barycenter_flag,
                   acceleration_settings_on_vehicle,
                   integrator_settings
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
        if self.Jupiter_barycenter_flag:
            body_settings.add_empty_settings("Jupiter_barycenter")
            body_settings.get("Jupiter_barycenter").gravity_field_settings = environment_setup.gravity_field.central(
                spice_interface.get_body_gravitational_parameter("5")
            )
            body_settings.get("Jupiter_barycenter").ephemeris_settings = environment_setup.ephemeris.direct_spice(
                body_name_to_use="5"
            )

        # Set gravity field settings for Enceladus
        body_settings.get("Enceladus").gravity_field_settings = Util.get_gravity_field_settings_enceladus_park()
        body_settings.get(
            "Enceladus").gravity_field_settings.scaled_mean_moment_of_inertia = 0.335  # From Iess et al. (2014)

        # Set gravity field settings for Saturn
        body_settings.get("Saturn").gravity_field_settings = Util.get_gravity_field_settings_saturn_iess()
        body_settings.get(
            "Saturn").gravity_field_settings.scaled_mean_moment_of_inertia = 0.210  # From NASA Saturn Fact Sheet

        # Define bodies that are propagated and their central bodies of propagation
        bodies_to_propagate = ["Vehicle"]
        central_bodies = ["Enceladus"]

        # Create global accelerations dictionary
        acceleration_settings = {"Vehicle": self.acceleration_settings_on_vehicle}

        dependent_variables_to_save = [
            propagation_setup.dependent_variable.relative_distance("Vehicle", "Enceladus"),
            propagation_setup.dependent_variable.altitude("Vehicle", "Enceladus"),
        ]

        [state_history, dependent_variables_history, computational_time] = self.propagate_orbit(initial_state,
                                                                                                body_settings,
                                                                                                bodies_to_propagate,
                                                                                                central_bodies,
                                                                                                acceleration_settings,
                                                                                                self.integrator_settings,
                                                                                                dependent_variables_to_save)

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
        occulting_bodies = {"Sun": ["Enceladus"]}
        vehicle_srp_settings = numerical_simulation.environment_setup.radiation_pressure.cannonball_radiation_target(
            VehicleParam.radiation_pressure_reference_area, VehicleParam.radiation_pressure_coefficient, occulting_bodies
        )
        environment_setup.add_radiation_pressure_target_model(bodies, "Vehicle", vehicle_srp_settings)

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
