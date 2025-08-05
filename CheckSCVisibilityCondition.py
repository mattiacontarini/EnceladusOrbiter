# Tudat import
from matplotlib import pyplot as plt
from tudatpy.math import interpolators
from tudatpy import constants

# Files import
from auxiliary import CovarianceAnalysisConfig as CovAnalysisConfig
from auxiliary.utilities import utilities

# General imports
import numpy as np
import os

# Define input path
input_directory = "./output/covariance_analysis/single_case_analysis"
time_stamp_folder = "2025.08.05.17.43.41"
input_path = os.path.join(input_directory, time_stamp_folder)
simulation_results_path = os.path.join(input_path, "simulation_results")
plots_path = os.path.join(input_path, "plots")
os.makedirs(plots_path, exist_ok=True)

# Load elevation angle history
for lander in CovAnalysisConfig.lander_names:
    target_angles_and_range = np.loadtxt(os.path.join(simulation_results_path, f"target_angles_and_range_{lander}.dat"))
    epochs = target_angles_and_range[:, 0]
    elevation_angle = target_angles_and_range[:, 1]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(epochs / constants.JULIAN_DAY, np.rad2deg(elevation_angle), marker=".", color="black")
    ax.axhline(y=np.rad2deg(CovAnalysisConfig.minimum_elevation_angle_visibility),
               color="red")
    ax.set_xlabel(r"$t - t_0$  [days]")
    ax.set_ylabel(r"$\delta$  [deg]")
    ax.grid(True)
    fig.savefig(os.path.join(plots_path, f"elevation_angle_{lander}.pdf"))
    plt.close(fig)
