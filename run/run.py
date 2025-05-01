import os
import sys
import time

os.environ["OPENBLAS_NUM_THREADS"] = "8"
sys.path.append(os.path.join(os.getcwd()))

import pandas as pd
import numpy as np

import llt_beam
# -----------------------------------------
# parameters
# -----------------------------------------

# wing path
wing_path = "wing_data/wing.csv"

# path to the folder where the experimentation will be saved
path_experimentation = "../data/LltBeam_wing_simulation/"  # replace with your path


# path to the folder containing the flight data
path_fp_folder = path_experimentation + f"flight_data/" # replace with your path

# create folder wing_and_simulation_characteristics if it does not exist
path_experimentation_wing_and_simulation_characteristics = (
    path_experimentation + "wing_and_simulation_characteristics/"
)
if not os.path.exists(path_experimentation_wing_and_simulation_characteristics):
    os.makedirs(path_experimentation_wing_and_simulation_characteristics)
    
# wing data
wing = pd.read_csv(wing_path, index_col=0)

# save the wing data
wing.to_csv(path_experimentation_wing_and_simulation_characteristics + "wing.csv")

# rayleigh parameters:
alpha_Rayleigh = np.load("wing_data/alpha_Rayleigh.npy")
beta_Rayleigh = np.load("wing_data/beta_Rayleigh.npy")

# aero matrices
K_aero = np.load("wing_data/K_aero.npy")
C_aero = np.load("wing_data/C_aero.npy")

# frequency of the flight point:
frequency = 500
gamma_newmark= 1 / 2
beta_newmark=1 / 4

print(beta_newmark, gamma_newmark)
# resolution of the lifting line theory
nb_fourrier_coeff = 150

# downsampling factor for the newmark saved data.
downsampling_factor = 5

# add gravity
add_gravity = True

print("--- saving of the wing and llt parameters ---")

# -----------------------------------------
# save the llt parameters
# -----------------------------------------

sim_parameters = {
    "resolution_lifting_line_theory": nb_fourrier_coeff,
    "gamma_newmark": gamma_newmark,
    "beta_newmark": beta_newmark,
    "original_frequency_flight_point": frequency,
    "downsampling_factor": downsampling_factor,
}

pd.DataFrame(sim_parameters, index=[0]).to_csv(
    path_experimentation_wing_and_simulation_characteristics
    + "sim_parameters.csv"
)

# -----------------------------------------
# beam model
# -----------------------------------------

# get the elementary stiffness matrix function of the beam
kelemf = llt_beam.get_elementary_stiffness_matrix_function()

# get the elementary mass matrix function of the beam
melemf = llt_beam.get_elementary_mass_matrix_function()

# instantiate the beam model
The_beam = llt_beam.AssembleBeamMatrix(
    wing,
    stiffness_elementary_matrix_function=kelemf,
    mass_elementary_matrix_function=melemf,
)

# get the P matrix
P_matrix = llt_beam.get_P_matrix(The_beam)

# get the matrices of the beam fem model
K, M, Q = The_beam.K_M_Q()

B_global = llt_beam.elementary_B_matrix(The_beam) # 
B = (wing["thickness"].values / 2)[:, None] * (B_global)


# add the aero and rayleigh damping
C = alpha_Rayleigh * M + beta_Rayleigh * K
C = C + C_aero
K = K + K_aero

# fixed degrees of freedom
K[0:2, :] = 0
K[:, 0:2] = 0
C[:2, :] = 0
C[:, :2] = 0

M[1:, 0] = 0
M[0, 1:] = 0
M[2:, 1] = 0
M[1, 2:] = 0

# get the proper modes of the system
mode_frequency, Phi = llt_beam.get_proper_mode(K + K_aero, M)
mode_omega = 2 * np.pi * mode_frequency

# normalise the modes by the mass matrix
Phi = llt_beam.mass_normalisation(M, Phi)

# get the strain modes
Psi = B @ Phi

# save the matrices for the beam model
np.save(path_experimentation_wing_and_simulation_characteristics + "K.npy", K)
np.save(path_experimentation_wing_and_simulation_characteristics + "M.npy", M)
np.save(path_experimentation_wing_and_simulation_characteristics + "C.npy", C)
np.save(path_experimentation_wing_and_simulation_characteristics + "B.npy", B)
np.save(path_experimentation_wing_and_simulation_characteristics + "P_matrix.npy", P_matrix)
np.save(path_experimentation_wing_and_simulation_characteristics + "K_aero.npy", K_aero)
np.save(path_experimentation_wing_and_simulation_characteristics + "C_aero.npy", C_aero)
np.save(path_experimentation_wing_and_simulation_characteristics + "alpha_Rayleigh.npy", alpha_Rayleigh)
np.save(path_experimentation_wing_and_simulation_characteristics + "beta_Rayleigh.npy", beta_Rayleigh)
np.save(path_experimentation_wing_and_simulation_characteristics + "Psi.npy", Psi)
np.save(path_experimentation_wing_and_simulation_characteristics + "Phi.npy", Phi)
np.save(path_experimentation_wing_and_simulation_characteristics + "mode_frequency.npy", mode_frequency)
np.save(path_experimentation_wing_and_simulation_characteristics + "mode_omega.npy", mode_omega)

# -----------------------------------------
# llt beam computation
# -----------------------------------------

# instantiate the llt beam
mylltbeam = llt_beam.LltBeam(
    wing,
    K,
    M,
    C,
    P_matrix,
    B=B,
    gamma_newmark=gamma_newmark,
    beta_newmark=beta_newmark,
    path_fp_folder=path_fp_folder,
    path_experimentation=path_experimentation,
    llt_resolution=nb_fourrier_coeff,
    frequency=frequency,
    downsampling_factor=downsampling_factor,
    add_gravity=add_gravity,
)

# run the simulation
mylltbeam.run()

