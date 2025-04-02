
# Files and variables import
from auxiliary import utilities
from auxiliary import VehicleParameters as VehicleParam

# Tudat import
from tudatpy.kernel.interface import spice_interface, spice
from tudatpy.numerical_simulation import environment_setup, propagation_setup

spice.load_standard_kernels()

def get_semimajor_axis_bounds(original_semimajor_axis, semiamplitude_variation_interval):
    return [original_semimajor_axis - semiamplitude_variation_interval, original_semimajor_axis + semiamplitude_variation_interval]


def get_escape_altitude(original_semimajor_axis, original_eccentricity, max_altitude_variation):
    apocentre_radius = original_semimajor_axis * (1 + original_eccentricity)
    enceladus_radius = spice_interface.get_average_radius("Enceladus")
    original_apocentre_altitude = apocentre_radius - enceladus_radius
    escape_altitude = original_apocentre_altitude + max_altitude_variation

    return escape_altitude


def retrieve_decision_variable_range(decision_variable_bounds_dict):
    lower_boundary = []
    upper_boundary = []
    for key in decision_variable_bounds_dict.keys():
        lower_boundary.append(decision_variable_bounds_dict[key][0])
        upper_boundary.append(decision_variable_bounds_dict[key][1])

    return (lower_boundary, upper_boundary)


def get_bodies(simulation_start_epoch):

    # Create settings for celestial bodies
    bodies_to_create = ["Sun",
                        "Saturn",
                        "Enceladus",
                        "Mimas",
                        "Tethys",
                        "Dione",
                        "Rhea",
                        "Titan",
                        "Hyperion"]
    global_frame_origin = "Enceladus"
    global_frame_orientation = "ECLIPJ2000"

    # Retrieve default body settings
    body_settings = environment_setup.get_default_body_settings(bodies_to_create,
                                                                global_frame_origin,
                                                                global_frame_orientation)

    # Add Jupiter barycenter
    body_settings.add_empty_settings("Jupiter_barycenter")
    body_settings.get("Jupiter_barycenter").gravity_field_settings = environment_setup.gravity_field.central(
        spice_interface.get_body_gravitational_parameter("5")
    )
    body_settings.get("Jupiter_barycenter").ephemeris_settings = environment_setup.ephemeris.direct_spice(
        body_name_to_use="5"
    )

    # Set rotation model settings for Enceladus
    synodic_rotation_rate_enceladus = utilities.get_synodic_rotation_model_enceladus(simulation_start_epoch)
    initial_orientation_enceladus = spice.compute_rotation_matrix_between_frames("ECLIPJ2000",
                                                                                 "IAU_Enceladus",
                                                                                 simulation_start_epoch)
    body_settings.get("Enceladus").rotation_model_settings = environment_setup.rotation_model.simple(
        "ECLIPJ2000", "IAU_Enceladus", initial_orientation_enceladus,
        simulation_start_epoch, synodic_rotation_rate_enceladus)

    # Set gravity field settings for Enceladus
    body_settings.get("Enceladus").gravity_field_settings = utilities.get_gravity_field_settings_enceladus_benedikter()

    # Set gravity field settings for Saturn
    body_settings.get("Saturn").gravity_field_settings = utilities.get_gravity_field_settings_saturn_benedikter()
    body_settings.get(
        "Saturn").gravity_field_settings.scaled_mean_moment_of_inertia = 0.210  # From NASA Saturn Fact Sheet

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

    return bodies


def get_bodies_to_propagate():
    return ["Vehicle"]

def get_central_bodies():
    return ["Enceladus"]


def get_acceleration_settings():

    acceleration_settings_on_vehicle = dict(
        Sun=[
            propagation_setup.acceleration.cannonball_radiation_pressure(),
            propagation_setup.acceleration.point_mass_gravity()
        ],
        Saturn=[
            propagation_setup.acceleration.spherical_harmonic_gravity(12, 12)
        ],
        Enceladus=[
            propagation_setup.acceleration.spherical_harmonic_gravity(3, 3)
        ],
        Jupiter_barycenter=[
            propagation_setup.acceleration.point_mass_gravity()
        ],
        Mimas=[
            propagation_setup.acceleration.point_mass_gravity()
        ],
        Tethys=[
            propagation_setup.acceleration.point_mass_gravity()
        ],
        Dione=[
            propagation_setup.acceleration.point_mass_gravity()
        ],
        Rhea=[
            propagation_setup.acceleration.point_mass_gravity()
        ],
        Titan=[
            propagation_setup.acceleration.point_mass_gravity()
        ],
        Hyperion=[
            propagation_setup.acceleration.point_mass_gravity()
        ]
    )

    acceleration_settings = {"Vehicle": acceleration_settings_on_vehicle}

    return acceleration_settings


def get_integrator_settings():

    fixed_step_size = 10
    integrator_settings = propagation_setup.integrator.runge_kutta_fixed_step(
        fixed_step_size,
        propagation_setup.integrator.CoefficientSets.rkf_78,
        order_to_use=propagation_setup.integrator.OrderToIntegrate.higher
    )
    return integrator_settings


def get_dependent_variables_to_save():
    dependent_variables_to_save = [
        propagation_setup.dependent_variable.relative_distance("Vehicle", "Enceladus")
    ]
    return dependent_variables_to_save
