# Thesis
Repository with the code and source data used for the thesis.

The code contained in this repository is used to perform the covariance analysis for a radio tracking science mission to Enceladus. The mission concept is composed of a low-altitude orbiter on a near-circular orbit, and of a set of radio beacons homogeneously spread on the surface of the moon. The spacecraft is connected through a two-way radio link to the radio beacons and to the Earth ground stations.

Three repeating orbits are available.

The following branches are available:
1. **propagator-selection**: the branch stores the scripts used to assess the integration error for a representative dynamical model.
2. **accelerations-selection**: the branch stores the scripts used to assess the physical model error.
3. **covariance-analysis**: the branch stores the scripts used to perform the covariance analysis for a single mission scenario, and offers the opportunity to explore the design space of interest and to generate figures of merit. The object that executes the covariance analysis is stored in the CovarianceAnalsysisObject.py file.
