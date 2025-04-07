# Tudat import
from tudatpy import constants
from tudatpy import numerical_simulation
from tudatpy.astro.time_conversion import DateTime

#######################################################################################################################
### Configuration #####################################################################################################
#######################################################################################################################

# Simulation epochs
simulation_start_epoch = DateTime(2000, 1, 1).epoch()  # From Benedikter et al. (2022)
simulation_duration = 20.52 * constants.JULIAN_DAY  # From Benedikter et al. (2022)
simulation_end_epoch = simulation_start_epoch + simulation_duration


#######################################################################################################################
### Environment #######################################################################################################
#######################################################################################################################

# Bodies included in the environment model
bodies_to_create = ["Sun",
                    "Saturn",
                    "Enceladus",
                    "Mimas",
                    "Tethys",
                    "Dione",
                    "Rhea"]

# Frame settings
global_frame_origin = "Enceladus"
global_frame_orientation = "J2000"

# Scaled mean moment of inertia of Enceladus and Saturn
Enceladus_scaled_mean_moment_of_inertia = 0.335  # From Iess et al. (2014)
Saturn_scaled_mean_moment_of_inertia = 0.210  # From NASA Saturn Fact Sheet

# Occulting bodies
occulting_bodies = dict()
occulting_bodies["Sun"] = ["Enceladus"]

#######################################################################################################################
### Propagation #######################################################################################################
#######################################################################################################################

# Define the accelerations acting on the vehicle
acceleration_settings_on_vehicle = dict(
    Sun=[
        numerical_simulation.propagation_setup.acceleration.radiation_pressure(),
        numerical_simulation.propagation_setup.acceleration.point_mass_gravity()
    ],
    Saturn=[
        numerical_simulation.propagation_setup.acceleration.spherical_harmonic_gravity(8, 8),
    ],
    Enceladus=[
        numerical_simulation.propagation_setup.acceleration.spherical_harmonic_gravity(3, 3),
        numerical_simulation.propagation_setup.acceleration.aerodynamic()
    ],
    Mimas=[
        numerical_simulation.propagation_setup.acceleration.point_mass_gravity()
    ],
    Tethys=[
        numerical_simulation.propagation_setup.acceleration.point_mass_gravity()
    ],
    Dione=[
        numerical_simulation.propagation_setup.acceleration.point_mass_gravity()
    ],
    Rhea=[
        numerical_simulation.propagation_setup.acceleration.point_mass_gravity()
    ]
)

# Define numerical integrator settings
fixed_step_size = 15
integrator_settings = numerical_simulation.propagation_setup.integrator.runge_kutta_fixed_step(
    fixed_step_size,
    numerical_simulation.propagation_setup.integrator.CoefficientSets.rkf_56,
    order_to_use=numerical_simulation.propagation_setup.integrator.OrderToIntegrate.higher
)

# Define propagation arcs during science phase
arc_duration = 1.0 * constants.JULIAN_DAY

