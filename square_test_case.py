import numpy as np
import porepy as pp

import discretizations
import problems
import simulators

##################################################
# Create grid
##################################################
# Create 2D domain with two fractures that cross
xmax = 3
ymax = 3
num_cells = [20, 20]

f1 = np.array([[0, xmax], [0.5 * ymax, 0.5 * ymax]])
f2 = np.array([[0.5 * xmax, 0.5 * xmax], [0, ymax]])

gb = pp.meshing.cart_grid([f1, f2], num_cells, physdims=[xmax, ymax])
# Uncomment for domain without fractures:
# gb = pp.meshing.cart_grid([], num_cells, physdims=[xmax, ymax])

##################################################
# Define wells
##################################################
# The wells are defined by the well centers: Well number i has
# coordinates well_pos[:, i].
well_pos = np.array([[2, 1, 2],
                     [2, 1, 1]])
# well_times[0, i] defines the time well i turns on and
# well_times[1, i] defines the time well i shuts down
well_times = np.array([[0, 1, 2],
                       [1, np.inf, np.inf]])
# well_rates[j, i] defines the rate [m^3 / s] of component j well i
source = np.array([1, 1, -1])
zer = np.zeros(well_pos.shape[1])

# well 1 inject component 0, well 2 inject component 1, well 3 produce
well_rates = np.array([[1, 0, -1],
                       [0, 1, 0]])

##################################################
# Set time stepping parameters
##################################################
time_step_param = {
    "initial_dt": 0.1 * pp.SECOND,
    "end_time": 4 * pp.SECOND,
    "max_dt": 0.1 * pp.SECOND,
    "vtk_folder_name": "res_temp/vtk",
    "file_name": "square",
}

##################################################
# Set parameter dictionary
##################################################
param = {
    "kf": 1,          # Fracture permeability
    "km": 1,          # Matrix permeability
    "kn": 1,          # Coupling law permeability (usually equal kf)
    "Df": 1,          # Diffusion in fracture
    "Dm": 1,          # diffusion in matrix
    "Dn": 1,          # Coupling law diffusion (usually equal Df)
    "aperture": 0.1,  # Fracture aperture
    "porosity": 1.0,  # Porosity
    "well_pos": well_pos,
    "well_times": well_times,
    "well_rates": well_rates,
    "time_step_param": time_step_param,
}

##################################################
# initiate problem and solve
##################################################
# Set up problem specific parameters
problem = problems.Fractured(gb, param)
# Define discretization
disc = discretizations.MiscibleFlow(problem)
# Run simulator
simulators.miscible(disc, problem)
